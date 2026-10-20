from sqlalchemy.orm import declarative_base
Base = declarative_base()


class Document(Base):
    __tablename__ = "docs"


def count_tags(conn):
    return conn.execute("""
        SELECT count(*) FROM tags t JOIN docs d ON d.id = t.doc_id
    """).fetchone()
