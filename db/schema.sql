-- lookup-agent database schema.
--
-- Safe to run again: it drops and recreates every table, so running it
-- wipes the data. Load it with:
--     python scripts/setup_db.py
-- or by hand:
--     docker compose exec -T postgres psql -U postgres -d lookup_db < db/schema.sql

CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS chunks, documents, support_tickets, order_items, orders, products, customers CASCADE;

-- ---------------------------------------------------------------------------
-- Shop tables (what the query_database tool reads)
-- ---------------------------------------------------------------------------

CREATE TABLE customers (
    id           INTEGER PRIMARY KEY,
    full_name    TEXT    NOT NULL,
    email        TEXT    NOT NULL UNIQUE,
    city         TEXT    NOT NULL,
    country      TEXT    NOT NULL,
    plan         TEXT    NOT NULL CHECK (plan IN ('basic', 'premium')),
    signup_date  DATE    NOT NULL
);

CREATE TABLE products (
    id        INTEGER        PRIMARY KEY,
    name      TEXT           NOT NULL UNIQUE,
    category  TEXT           NOT NULL,
    price     NUMERIC(10, 2) NOT NULL CHECK (price > 0)   -- USD
);

CREATE TABLE orders (
    id           INTEGER        PRIMARY KEY,
    customer_id  INTEGER        NOT NULL REFERENCES customers (id),
    order_date   DATE           NOT NULL,
    status       TEXT           NOT NULL CHECK (status IN ('delivered', 'processing', 'cancelled')),
    total        NUMERIC(10, 2) NOT NULL CHECK (total >= 0)  -- USD, sum of its items
);

CREATE TABLE order_items (
    id          INTEGER        PRIMARY KEY,
    order_id    INTEGER        NOT NULL REFERENCES orders (id) ON DELETE CASCADE,
    product_id  INTEGER        NOT NULL REFERENCES products (id),
    quantity    INTEGER        NOT NULL CHECK (quantity > 0),
    unit_price  NUMERIC(10, 2) NOT NULL CHECK (unit_price > 0)  -- price at the time of the order
);

CREATE TABLE support_tickets (
    id           INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers (id),
    order_id     INTEGER REFERENCES orders (id),   -- NULL when not about one order
    topic        TEXT    NOT NULL CHECK (topic IN ('delivery', 'refund', 'damaged_item', 'quality')),
    status       TEXT    NOT NULL CHECK (status IN ('open', 'closed')),
    created_at   DATE    NOT NULL,
    description  TEXT    NOT NULL
);

CREATE INDEX orders_customer_id_idx        ON orders (customer_id);
CREATE INDEX orders_order_date_idx         ON orders (order_date);
CREATE INDEX order_items_order_id_idx      ON order_items (order_id);
CREATE INDEX order_items_product_id_idx    ON order_items (product_id);
CREATE INDEX support_tickets_customer_idx  ON support_tickets (customer_id);
CREATE INDEX support_tickets_created_idx   ON support_tickets (created_at);

-- ---------------------------------------------------------------------------
-- Search tables (what the search_docs tool reads)
-- ---------------------------------------------------------------------------

CREATE TABLE documents (
    id            SERIAL      PRIMARY KEY,
    source        TEXT        NOT NULL UNIQUE,   -- file name, e.g. refunds.md
    title         TEXT        NOT NULL,
    content_hash  TEXT        NOT NULL,          -- sha256 of the file, to skip unchanged files
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id           SERIAL  PRIMARY KEY,
    document_id  INTEGER NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    heading      TEXT    NOT NULL,
    content      TEXT    NOT NULL,
    embedding    vector(384) NOT NULL,           -- must match EMBEDDING_DIM
    tsv          tsvector GENERATED ALWAYS AS (
                     to_tsvector('english', heading || ' ' || content)
                 ) STORED,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_idx       ON chunks USING gin (tsv);

-- ---------------------------------------------------------------------------
-- Read-only user for the agent's SQL tool.
-- It can SELECT from the shop tables and nothing else. Every transaction
-- is read-only and every query is stopped after 5 seconds.
-- The password is a local demo default; change it for a real deploy.
-- ---------------------------------------------------------------------------

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'lookup_reader') THEN
        CREATE ROLE lookup_reader LOGIN PASSWORD 'lookup_reader';
    END IF;
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO lookup_reader', current_database());
END
$$;

REVOKE ALL ON ALL TABLES IN SCHEMA public FROM lookup_reader;
GRANT USAGE ON SCHEMA public TO lookup_reader;
GRANT SELECT ON customers, products, orders, order_items, support_tickets TO lookup_reader;
ALTER ROLE lookup_reader SET default_transaction_read_only = on;
ALTER ROLE lookup_reader SET statement_timeout = '5s';
