use rag::llm::{LlmClient, TongyiClient};
use async_openai::types::{ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {

    // 创建通义千问客户端
    let client = TongyiClient::new()
        .with_model("qwen-max".to_string())
        .with_temperature(0.7)
        .with_max_tokens(2000);

    println!("🤖 通义千问聊天测试\n");

    let messages = vec![
        ChatCompletionRequestMessage::System(
            ChatCompletionRequestSystemMessageArgs::default()
                .content("你是一个知识渊博的AI助手。")
                .build()?
        ),
        ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content("Rust语言的主要特点是什么？请简要说明。")
                .build()?
        ),
    ];

    match client.chat(messages).await {
        Ok(response) => {
            println!("✅ 回复: {}\n", response);
        }
        Err(e) => {
            eprintln!("❌ 错误: {}\n", e);
        }
    }

    println!("🎉 测试完成！");

    Ok(())
}
