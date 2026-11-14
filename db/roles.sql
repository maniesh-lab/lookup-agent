-- The read-only user the agent's SQL tool connects as (lookup_reader).
--
-- It can SELECT from the five shop tables and nothing else. Every
-- transaction is read-only and every query is stopped after 5 seconds.
-- The password is a local demo default; change it for a real deploy.
--
-- Safe to run any number of times, and it doesn't touch any data:
-- scripts.setup_db runs it on every start, so a rule added here also
-- reaches a database that already exists. Needs the tables from
-- db/schema.sql, and a superuser for the two REVOKE lines at the end.
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

-- Every role may cancel or end the queries of other sessions of the same role,
-- so one agent query could stop another user's. Only the owner (a superuser,
-- not affected by this) needs these functions here.
REVOKE EXECUTE ON FUNCTION pg_catalog.pg_cancel_backend(integer) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION pg_catalog.pg_terminate_backend(integer, bigint) FROM PUBLIC;
