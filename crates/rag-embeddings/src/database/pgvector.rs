use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::Utc;
use sqlx::PgPool;
use uuid::Uuid;

use crate::database::{VectorRecord, VectorStore};

pub struct PgVectorStore {
    pool: PgPool,
    table_name: String,
    dimensions: usize,
}

impl PgVectorStore {
    pub async fn new(pool: PgPool, table_name: &str, dimensions: usize) -> Result<Self> {
        let store = Self {
            pool,
            table_name: table_name.to_string(),
            dimensions,
        };
        store.init_table().await?;
        Ok(store)
    }

    async fn init_table(&self) -> Result<()> {

        sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
            .execute(&self.pool)
            .await
            .context("Failed to create vector extension")?;

        let sql = format!(
            r#"
            CREATE TABLE IF NOT EXISTS {} (
                id UUID PRIMARY KEY,
                embedding VECTOR({}),
                metadata JSONB DEFAULT '{{}}'::jsonb,
                text TEXT,
                createat TIMESTAMPTZ DEFAULT NOW(),
                updateat TIMESTAMPTZ DEFAULT NOW()
            );"#,
            self.table_name,
            self.dimensions,
        );
        
        sqlx::query(&sql)
            .execute(&self.pool)
            .await
            .context("Failed to init vector table")?;
        
        Ok(())
    }

}

#[async_trait]
impl VectorStore for PgVectorStore {
    async fn add_vectors(&self, vectors: Vec<VectorRecord>) -> Result<()> {
        let mut tx = self.pool.begin().await?;

        for vec in vectors {
            let id = Uuid::parse_str(&vec.id)
                .context(format!("Invalid UUID: {}", vec.id))?;
            if vec.embedding.len() != self.dimensions {
                anyhow::bail!(
                    "Embedding dim mismatch: expected {}, got {}",
                    self.dimensions,
                    vec.embedding.len()
                );
            }
            let now = Utc::now();
            let createat = vec.createat.unwrap_or(now);
            let updateat = vec.updateat.unwrap_or(now);

            sqlx::query(&format!(
                r#"INSERT INTO "{}" (id, embedding, metadata, text, createat, updateat) 
                   VALUES ($1, $2, $3, $4, $5, $6)"#,
                self.table_name
            ))
            .bind(id)
            .bind(&vec.embedding)
            .bind(&vec.metadata)
            .bind(&vec.text)
            .bind(createat)
            .bind(updateat)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    async fn upsert_vectors(&self, vectors: Vec<VectorRecord>) -> Result<()> {
        let mut tx = self.pool.begin().await?;

        for vec in vectors {
            let id = Uuid::parse_str(&vec.id)?;
            if vec.embedding.len() != self.dimensions {
                continue;
            }
            let now = Utc::now();
            let createat = vec.createat.unwrap_or(now);
            let updateat = vec.updateat.unwrap_or(now);

            sqlx::query(&format!(
                r#"INSERT INTO "{}" (id, embedding, metadata, text, createat, updateat)
                   VALUES ($1, $2, $3, $4, $5, $6)
                   ON CONFLICT (id) DO UPDATE SET
                     embedding = EXCLUDED.embedding,
                     metadata = EXCLUDED.metadata,
                     text = EXCLUDED.text,
                     updateat = EXCLUDED.updateat"#,
                self.table_name
            ))
            .bind(id)
            .bind(&vec.embedding)
            .bind(&vec.metadata)
            .bind(&vec.text)
            .bind(createat)
            .bind(updateat)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    async fn delete_vector(&self, ids: Vec<String>) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }

        let placeholders = (1..=ids.len()).map(|i| format!("${}", i)).collect::<Vec<_>>();
        let sql = format!(
            r#"DELETE FROM "{}" WHERE id IN ({})"#,
            self.table_name,
            placeholders.join(", ")
        );

        let mut query = sqlx::query(&sql);
        for id_str in ids {
            let uuid = Uuid::parse_str(&id_str)?;
            query = query.bind(uuid);
        }

        query.execute(&self.pool).await?;
        Ok(())
    }

    async fn search(&self) -> Result<Vec<VectorRecord>> {
        let rows = sqlx::query_as::<_, VectorRecord>(&format!(
            r#"SELECT id::text, embedding, metadata, text, createat, updateat 
               FROM "{}""#,
            self.table_name
        ))
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::postgres::PgPoolOptions;
    #[tokio::test]
    async fn test_add_vector() { 
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect("postgres:///rag_db")
            .await
            .expect("Failed to connect");

        let store = PgVectorStore::new(pool,"test1",3)
            .await
            .expect("Failed to create PgvectorStore");

        let record = VectorRecord {
            id: "00000000-0000-0000-0000-000000000001".to_string(),
            embedding: vec![1.0, 2.0, 3.0],
            metadata: serde_json::json!({}),
            text: Some("text".to_string()),
            createat: Some(Utc::now()),
            updateat: Some(Utc::now()),
        };


        store.add_vectors(vec![record]).await.unwrap();
        println!("Added vector")
    }

    #[tokio::test]
    async fn delete_vector() {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect("postgres:///rag_db")
            .await
            .expect("failed to connect");

        let store = PgVectorStore::new(pool,"test1",3)
            .await
            .expect("Faile to create Pgstore");

        let maybe = store.delete_vector(vec!["00000000-0000-0000-0000-000000000001".to_string()]).await.unwrap();
        println!("maybe: {:?}",maybe);
    }
}