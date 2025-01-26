import { OpenAI } from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';
import { streamText } from 'ai';
import { openai } from '@ai-sdk/openai';

// Initialize clients
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY!,
});

const openaiClient = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

const index = pinecone.Index(process.env.PINECONE_INDEX_NAME!);

// Improved content retrieval with better context management
async function getRelevantContent(query: string, numResults: number = 5) {
  const queryEmbedding = await openaiClient.embeddings.create({
    model: "text-embedding-ada-002",
    input: query,
  });

  const searchResults = await index.query({
    vector: queryEmbedding.data[0].embedding,
    topK: numResults * 2, // Fetch more results initially for better filtering
    includeMetadata: true,
  });

  // Filter and sort results by relevance score
  const filteredResults = searchResults.matches
    ?.filter(match => match.score && match.score > 0.7) // Adjust threshold as needed
    .slice(0, numResults);

  return filteredResults;
}

// Format context with better structure and metadata
function formatContext(posts: any[]) {
  return posts
    .filter(post => post.metadata?.link && post.metadata?.title)
    .map(post => `
ARTICLE:
Title: ${post.metadata.title}
URL: ${post.metadata.link}
Key Points: ${post.metadata?.excerpt || 'No excerpt available'}
Relevance Score: ${post.score?.toFixed(2) || 'N/A'}
---`)
    .join('\n\n');
}

export async function POST(req: Request) {
  const { messages } = await req.json();
  const userQuery = messages[messages.length - 1].content;

  try {
    const relevantPosts = await getRelevantContent(userQuery);

    if (!relevantPosts?.length) {
      return streamText({
        model: openai('gpt-4o-mini'),
        messages: [
          { 
            role: 'system', 
            content: "You are a helpful chatbot for SteveBizBlog that provides specific, focused answers." 
          },
          { 
            role: 'user', 
            content: userQuery 
          },
          { 
            role: 'assistant', 
            content: "I couldn't find specific information about this in Steve's blog posts. Could you rephrase your question or ask about a related business topic?" 
          },
        ],
      }).toDataStreamResponse();
    }

    const { formattedContent, urls } = formatContext(relevantPosts);

    const systemPrompt = `You are "SteveBizBot", a focused and concise chatbot for SteveBizBlog.com. You MUST follow these guidelines:

1. Start your response with a clear, focused answer
2. After your main response, you MUST include a "References:" section with 2-5 relevant URLs from the provided content
3. Format each reference as a bullet point with the title and URL
4. Keep responses focused and under 3-4 paragraphs
5. If information is partial, acknowledge what's known from the blog and what's not
6. Maintain a professional but conversational tone

Your response MUST follow this format:
[Main response content]

References:
• [Title 1] - [URL 1]
• [Title 2] - [URL 2]
[etc.]

Available source material for this query:
${formattedContent}

Remember: ALWAYS include the References section with URLs at the end of your response.`;

    return streamText({
      model: openai('gpt-4o-mini'),
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'system', content: `Reference Content:\n${formattedContent}` },
        ...messages.slice(-3), // Only include last 3 messages for context
      ],
      max_tokens: 8000, // Reduced from 16000 for more focused responses
      temperature: 0.2, // Slightly increased for better articulation while maintaining consistency
      presence_penalty: 0.3, // Encourage some variety in responses
      frequency_penalty: 0.3, // Discourage repetitive language
    }).toDataStreamResponse();
  } catch (error) {
    console.error('Error processing request:', error);
    throw new Error('Failed to process request');
  }
}
