use async_openai::types::ChatCompletionRequestMessage;
use async_trait::async_trait;
use anyhow::Result;

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn chat(&self, messages: Vec<ChatCompletionRequestMessage>) -> Result<String>;

    async fn generate(&self, messages: Vec<ChatCompletionRequestMessage>) -> Result<String>;

}