"""SQLModel job/session/turn/reply store and per-job flock coordination."""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from sqlalchemy import Connection, Engine, event
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import URL
from sqlalchemy.orm import registry
from sqlmodel import Field, MetaData, Session, SQLModel, col, create_engine, or_, select, update
from sqlmodel.sql.sqltypes import AutoString

from castle_steward_engine.workflow import AgentSpec

TurnStatus = Literal["reserved", "running", "completed", "failed", "cancelled"]

_DEFAULT_STATE_DIR_NAME = "castle-steward-engine"
_DB_FILENAME = "engine.db"
_LOCKS_DIRNAME = "locks"
_SESSIONS_DIRNAME = "sessions"
_ARTIFACTS_DIRNAME = "artifacts"
SQLITE_BUSY_TIMEOUT_SECONDS = 5.0
_LOCK_FILE_MODE = 0o600
_JOB_LOCK_RETRY_ATTEMPTS = 10
_JOB_LOCK_RETRY_DELAY_SECONDS = 0.005

_NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class _StoreBase(
    SQLModel,
    registry=registry(metadata=MetaData(naming_convention=_NAMING_CONVENTION)),
):
    pass


METADATA = _StoreBase.metadata


class JobRecord(_StoreBase, table=True):
    __tablename__ = "jobs"

    job_id: UUID = Field(primary_key=True)
    workflow: str
    workflow_version: str
    input_json: str
    last_error: str | None = None
    stop_seq: int = 0
    ack_seq: int = 0
    created_at: str
    updated_at: str

    @property
    def stop_pending(self) -> bool:
        return self.stop_seq > self.ack_seq


class ReplyRecord(_StoreBase, table=True):
    __tablename__ = "replies"

    job_id: UUID = Field(primary_key=True)
    wait_id: str = Field(primary_key=True)
    reply_json: str
    answered_at: str


class SessionRecord(_StoreBase, table=True):
    __tablename__ = "sessions"

    job_id: UUID = Field(primary_key=True)
    session_name: str = Field(primary_key=True)
    session_id: str
    role_name: str
    al_handle: str
    agent_json: str
    workdir: str
    created_at: str

    @property
    def agent(self) -> AgentSpec:
        return AgentSpec.model_validate_json(self.agent_json)

    @property
    def workdir_path(self) -> Path:
        return Path(self.workdir)


class TurnRecord(_StoreBase, table=True):
    __tablename__ = "turns"

    job_id: UUID = Field(primary_key=True)
    turn_key: str = Field(primary_key=True)
    attempt: int = Field(primary_key=True)
    node: str
    session_name: str
    al_handle: str
    al_invocation_id: str
    status: TurnStatus = Field(sa_type=AutoString)
    prompt_path: str
    created_at: str
    completed_at: str | None = None

    @property
    def prompt_file(self) -> Path:
        return Path(self.prompt_path)


INCLUDE_TABLES: frozenset[str] = frozenset(METADATA.tables)


def include_name(name: str | None, type_: str, parent_names: Mapping[Any, str | None]) -> bool:
    """Filter Alembic reflection strictly to engine-owned tables (ignoring LangGraph tables)."""
    del parent_names
    return type_ != "table" or name in INCLUDE_TABLES


def default_state_dir() -> Path:
    """Return default engine state directory (~/.local/state/castle-steward-engine)."""
    xdg = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "state"
    return base / _DEFAULT_STATE_DIR_NAME


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _create_engine(url: URL) -> Engine:
    connect_args = (
        {"timeout": SQLITE_BUSY_TIMEOUT_SECONDS} if url.get_backend_name() == "sqlite" else {}
    )
    engine = create_engine(url, connect_args=connect_args)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _on_sqlite_connect(dbapi_connection: Any, _connection_record: Any) -> None:
            dbapi_connection.isolation_level = None
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

        @event.listens_for(engine, "begin")
        def _on_sqlite_begin(conn: Connection) -> None:
            if conn.get_execution_options().get("isolation_level") == "AUTOCOMMIT":
                return
            conn.exec_driver_sql("BEGIN IMMEDIATE")

    return engine


class JobBusy(RuntimeError):
    """Raised when another runner already holds the exclusive lock for a job."""


class JobNotFoundError(KeyError):
    """Raised when a job ID does not exist in the store."""


class Store:
    """Owns SQLModel engine metadata (separate from SqliteSaver) and per-job flock locks."""

    def __init__(self, state_dir: Path | None = None) -> None:
        self.state_dir = (state_dir or default_state_dir()).resolve()
        self.db_path = self.state_dir / _DB_FILENAME
        self.locks_dir = self.state_dir / _LOCKS_DIRNAME
        self.sessions_dir = self.state_dir / _SESSIONS_DIRNAME
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._engine = _create_engine(URL.create("sqlite", database=str(self.db_path)))
        self._read_engine = self._engine.execution_options(isolation_level="AUTOCOMMIT")
        try:
            self._migrate()
        except BaseException:
            self._engine.dispose()
            raise

    def session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def artifacts_dir(self, job_id: UUID) -> Path:
        path = self.state_dir / _ARTIFACTS_DIRNAME / str(job_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def close(self) -> None:
        self._engine.dispose()

    def _write_session(self) -> Session:
        return Session(self._engine, expire_on_commit=False)

    def _read_session(self) -> Session:
        return Session(self._read_engine)

    def _migrate(self) -> None:
        from alembic import command
        from alembic.config import Config
        from alembic.runtime.migration import MigrationContext
        from alembic.script import ScriptDirectory

        cfg = Config()
        cfg.set_main_option("script_location", str(Path(__file__).resolve().parent / "migrations"))
        script_head = ScriptDirectory.from_config(cfg).get_current_head()
        if script_head is None:
            raise RuntimeError("Alembic migrations directory has no head revision")

        with self._read_engine.connect() as read_conn:
            if MigrationContext.configure(read_conn).get_current_revision() == script_head:
                return

        with self._engine.begin() as conn:
            cfg.attributes["connection"] = conn
            command.upgrade(cfg, "head")
            current = MigrationContext.configure(conn).get_current_revision()
            if current != script_head:
                raise RuntimeError(
                    f"Database migration left revision at {current!r}, expected {script_head!r}"
                )

    @contextmanager
    def job_lock(self, job_id: UUID) -> Iterator[None]:
        """Acquire a non-blocking exclusive flock for `job_id`, raising JobBusy if held."""
        fd = os.open(self.locks_dir / f"{job_id}.lock", os.O_CREAT | os.O_RDWR, _LOCK_FILE_MODE)
        try:
            for attempt in range(_JOB_LOCK_RETRY_ATTEMPTS):
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if attempt + 1 == _JOB_LOCK_RETRY_ATTEMPTS:
                        raise JobBusy(f"Job {job_id} is already running") from exc
                    time.sleep(_JOB_LOCK_RETRY_DELAY_SECONDS)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def is_running(self, job_id: UUID) -> bool:
        """Non-blocking probe via LOCK_SH | LOCK_NB to check if a job is currently running."""
        path = self.locks_dir / f"{job_id}.lock"
        if not path.exists():
            return False
        fd = os.open(path, os.O_RDONLY)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        finally:
            os.close(fd)

    def create_job(
        self, job_id: UUID, *, workflow: str, workflow_version: str, input_json: str
    ) -> JobRecord:
        now = _utc_now()
        job = JobRecord(
            job_id=job_id,
            workflow=workflow,
            workflow_version=workflow_version,
            input_json=input_json,
            created_at=now,
            updated_at=now,
        )
        with self._write_session() as session:
            session.add(job)
            session.commit()
        return job

    def get_job(self, job_id: UUID) -> JobRecord:
        with self._read_session() as session:
            job = session.get(JobRecord, job_id)
        if job is None:
            raise JobNotFoundError(f"Unknown job: {job_id}")
        return job

    def request_stop(self, job_id: UUID) -> None:
        with self._write_session() as session:
            result = session.exec(
                update(JobRecord)
                .where(col(JobRecord.job_id) == job_id)
                .values(stop_seq=col(JobRecord.stop_seq) + 1, updated_at=_utc_now())
            )
            if result.rowcount == 0:
                raise JobNotFoundError(f"Unknown job: {job_id}")
            session.commit()

    def is_stop_pending(self, job_id: UUID) -> bool:
        return self.get_job(job_id).stop_pending

    def ack_stop(self, job_id: UUID) -> None:
        with self._write_session() as session:
            session.exec(
                update(JobRecord)
                .where(
                    col(JobRecord.job_id) == job_id,
                    col(JobRecord.stop_seq) > col(JobRecord.ack_seq),
                )
                .values(ack_seq=col(JobRecord.stop_seq), updated_at=_utc_now())
            )
            session.commit()

    def set_last_error(self, job_id: UUID, error: str | None) -> None:
        with self._write_session() as session:
            job = session.get(JobRecord, job_id)
            if job is None:
                raise JobNotFoundError(f"Unknown job: {job_id}")
            job.last_error = error
            job.updated_at = _utc_now()
            session.commit()

    def record_reply(self, job_id: UUID, wait_id: str, reply_json: str) -> None:
        with self._engine.begin() as conn:
            if conn.dialect.name != "sqlite":
                raise NotImplementedError(f"Reply upsert unsupported for {conn.dialect.name!r}")
            stmt = sqlite_insert(ReplyRecord).values(
                job_id=job_id, wait_id=wait_id, reply_json=reply_json, answered_at=_utc_now()
            )
            conn.execute(
                stmt.on_conflict_do_update(
                    index_elements=["job_id", "wait_id"],
                    set_={
                        "reply_json": stmt.excluded.reply_json,
                        "answered_at": stmt.excluded.answered_at,
                    },
                )
            )

    def get_replies(self, job_id: UUID, wait_ids: Sequence[str]) -> dict[str, str]:
        if not wait_ids:
            return {}
        with self._read_session() as session:
            rows = session.exec(
                select(ReplyRecord.wait_id, ReplyRecord.reply_json).where(
                    col(ReplyRecord.job_id) == job_id,
                    col(ReplyRecord.wait_id).in_(wait_ids),
                )
            ).all()
        return dict(rows)

    def get_session(self, job_id: UUID, session_name: str) -> SessionRecord | None:
        with self._read_session() as session:
            return session.get(SessionRecord, (job_id, session_name))

    def get_latest_turn(self, job_id: UUID, turn_key: str) -> TurnRecord | None:
        with self._read_session() as session:
            return session.exec(
                select(TurnRecord)
                .where(col(TurnRecord.job_id) == job_id, col(TurnRecord.turn_key) == turn_key)
                .order_by(col(TurnRecord.attempt).desc())
                .limit(1)
            ).one_or_none()

    def list_active_turns(self, job_id: UUID) -> list[TurnRecord]:
        with self._read_session() as session:
            return list(
                session.exec(
                    select(TurnRecord)
                    .where(
                        col(TurnRecord.job_id) == job_id,
                        col(TurnRecord.status).in_(("reserved", "running")),
                    )
                    .order_by(col(TurnRecord.created_at).asc(), col(TurnRecord.attempt).asc())
                ).all()
            )

    def is_initial_session_turn(
        self, job_id: UUID, al_handle: str, turn_key: str, attempt: int
    ) -> bool:
        """True iff no other turn row shares this Agent Layer handle."""
        stmt = (
            select(TurnRecord.job_id)
            .where(
                col(TurnRecord.job_id) == job_id,
                col(TurnRecord.al_handle) == al_handle,
                or_(col(TurnRecord.turn_key) != turn_key, col(TurnRecord.attempt) != attempt),
            )
            .limit(1)
        )
        with self._read_session() as session:
            return session.exec(stmt).first() is None

    def is_first_turn_of_session(self, job_id: UUID, session_name: str, turn_key: str) -> bool:
        """True iff this session has no completed turns and no other turn_key."""
        stmt = (
            select(TurnRecord.job_id)
            .where(
                col(TurnRecord.job_id) == job_id,
                col(TurnRecord.session_name) == session_name,
                or_(col(TurnRecord.turn_key) != turn_key, col(TurnRecord.status) == "completed"),
            )
            .limit(1)
        )
        with self._read_session() as session:
            return session.exec(stmt).first() is None

    def create_session_and_turn(self, session: SessionRecord, turn: TurnRecord) -> None:
        """Atomically record a newly reserved session and its first turn."""
        with self._write_session() as db:
            db.add_all((session, turn))
            db.commit()

    def update_session_handle_and_create_turn(
        self, job_id: UUID, session_name: str, al_handle: str, turn: TurnRecord
    ) -> SessionRecord:
        """Atomically replace a dead initial handle (keeping session_id immutable) and add turn."""
        with self._write_session() as db:
            rec = db.get(SessionRecord, (job_id, session_name))
            if rec is None:
                raise LookupError(f"Unknown session {session_name!r} for job {job_id}")
            rec.al_handle = al_handle
            db.add(turn)
            db.commit()
            return rec

    def create_turn(self, turn: TurnRecord) -> None:
        with self._write_session() as db:
            db.add(turn)
            db.commit()

    def update_turn_status(
        self,
        job_id: UUID,
        turn_key: str,
        attempt: int,
        status: TurnStatus,
        *,
        al_handle: str | None = None,
        al_invocation_id: str | None = None,
        completed: bool = False,
    ) -> TurnRecord:
        with self._write_session() as db:
            turn = db.get(TurnRecord, (job_id, turn_key, attempt))
            if turn is None:
                raise LookupError(f"Unknown turn {turn_key!r} (attempt {attempt}) for job {job_id}")
            turn.status = status
            if al_handle is not None:
                turn.al_handle = al_handle
            if al_invocation_id is not None:
                turn.al_invocation_id = al_invocation_id
            if completed:
                turn.completed_at = _utc_now()
            db.commit()
            return turn
