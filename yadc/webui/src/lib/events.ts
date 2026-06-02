import { type ZodType, ZodError } from 'zod';

type DataHandler<T> = (event: T) => void;
type ErrorHandler = (error: Error) => void;

/**
 * A typed EventSource wrapper that validates each SSE event against a Zod schema.
 * Supports listening to multiple named event types with independent schemas.
 */
export class TypedEventSource extends EventSource {
	listen<T>(
		eventType: string,
		schema: ZodType<T>,
		onData: DataHandler<T>,
		onError?: ErrorHandler
	): void {
		const handleError =
			onError || ((err) => console.error(`Error in ${eventType} SSE event handler:`, err));

		this.addEventListener(eventType, (rawEvent: MessageEvent) => {
			try {
				const rawData = JSON.parse(rawEvent.data);
				const parsedData = schema.parse(rawData);

				onData(parsedData);
			} catch (err) {
				if (err instanceof ZodError) {
					handleError(new Error(`Validation failed for event "${eventType}": ${err.message}`));
				} else if (err instanceof SyntaxError) {
					handleError(new Error(`Invalid JSON in event "${eventType}": ${err.message}`));
				} else {
					handleError(err as Error);
				}
			}
		});
	}
}
