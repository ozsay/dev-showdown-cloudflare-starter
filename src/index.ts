import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { Output, generateText, stepCountIs, streamText, tool } from 'ai';
import { z } from 'zod';

const INTERACTION_ID_HEADER = 'X-Interaction-Id';

type WorkshopLlm = ReturnType<typeof createWorkshopLlm>;

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const url = new URL(request.url);

		if (request.method !== 'POST' || url.pathname !== '/api') {
			return new Response('Not Found', { status: 404 });
		}

		const challengeType = url.searchParams.get('challengeType');
		if (!challengeType) {
			return new Response('Missing challengeType query parameter', {
				status: 400,
			});
		}

		const interactionId = request.headers.get(INTERACTION_ID_HEADER);
		if (!interactionId) {
			return new Response(`Missing ${INTERACTION_ID_HEADER} header`, {
				status: 400,
			});
		}

		if (!env.DEV_SHOWDOWN_API_KEY) {
			throw new Error('DEV_SHOWDOWN_API_KEY is required');
		}

		const payload = await request.json<any>();
		const workshopLlm = createWorkshopLlm(env.DEV_SHOWDOWN_API_KEY, interactionId);

		switch (challengeType) {
			case 'HELLO_WORLD':
				return helloWorld(workshopLlm, payload);
			case 'BASIC_LLM':
				return basicLlm(workshopLlm, payload);
			case 'JSON_MODE':
				return jsonMode(workshopLlm, payload);
			case 'BASIC_TOOL_CALL':
				return basicToolCall(workshopLlm, payload, interactionId);
			case 'RESPONSE_STREAMING':
				return responseStreaming(workshopLlm, payload);
			default:
				return new Response('Solver not found', { status: 404 });
		}
	},
} satisfies ExportedHandler<Env>;

async function helloWorld(_workshopLlm: WorkshopLlm, payload: any): Promise<Response> {
	return Response.json({
		greeting: `Hello ${payload.name}`,
	});
}

async function basicLlm(workshopLlm: WorkshopLlm, payload: any): Promise<Response> {
	const result = await generateText({
		model: workshopLlm.chatModel('deli-4'),
		system: 'You are a trivia question player. Answer the question correctly and concisely.',
		prompt: payload.question,
	});

	return Response.json({
		answer: result.text || 'N/A',
	});
}

const productSchema = z.object({
	name: z.string(),
	price: z.number(),
	currency: z.string(),
	inStock: z.boolean(),
	dimensions: z.object({
		length: z.number(),
		width: z.number(),
		height: z.number(),
		unit: z.string(),
	}),
	manufacturer: z.object({
		name: z.string(),
		country: z.string(),
		website: z.string(),
	}),
	specifications: z.object({
		weight: z.number(),
		weightUnit: z.string(),
		warrantyMonths: z.number(),
	}),
});

async function basicToolCall(workshopLlm: WorkshopLlm, payload: any, interactionId: string): Promise<Response> {
	const result = await generateText({
		model: workshopLlm.chatModel('deli-4'),
		system:
			'You answer weather questions. Use the getWeather tool to look up the temperature for the city in the question, then reply in one short natural-language sentence that includes the temperature.',
		prompt: payload.question,
		stopWhen: stepCountIs(5),
		tools: {
			getWeather: tool({
				description: 'Get the current weather for a city.',
				inputSchema: z.object({
					city: z.string().describe('The city to get the weather for'),
				}),
				execute: async ({ city }) => {
					const res = await fetch('https://devshowdown.com/api/weather', {
						method: 'POST',
						headers: {
							'Content-Type': 'application/json',
							[INTERACTION_ID_HEADER]: interactionId,
						},
						body: JSON.stringify({ city }),
					});
					return await res.json();
				},
			}),
		},
	});

	return Response.json({
		answer: result.text || 'N/A',
	});
}

async function responseStreaming(workshopLlm: WorkshopLlm, payload: any): Promise<Response> {
	const result = streamText({
		model: workshopLlm.chatModel('deli-4'),
		prompt: payload.prompt,
	});

	return result.toTextStreamResponse();
}

async function jsonMode(workshopLlm: WorkshopLlm, payload: any): Promise<Response> {
	const result = await generateText({
		model: workshopLlm.chatModel('deli-4'),
		system:
			'You extract structured product data from a free-form description. Every required fact is present in the text.',
		prompt: payload.description,
		output: Output.object({ schema: productSchema }),
	});

	return Response.json(result.output);
}

function createWorkshopLlm(apiKey: string, interactionId: string) {
	return createOpenAICompatible({
		name: 'dev-showdown',
		baseURL: 'https://devshowdown.com/v1',
		supportsStructuredOutputs: true,
		headers: {
			Authorization: `Bearer ${apiKey}`,
			[INTERACTION_ID_HEADER]: interactionId,
		},
	});
}
