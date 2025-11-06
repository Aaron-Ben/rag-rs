use sqlx::postgres::PgPoolOptions;
use anyhow::Result;


fn main() -> Result<()> {
    let _ = PgPoolOptions::new()
        .max_connections(5)
        .connect("postgres:///rag_db");
    println!("connected to database");
    Ok(())
}