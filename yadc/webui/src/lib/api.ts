/**
 * Base URL for API requests.
 * In production (Flask serves both), this is empty (same-origin).
 * In development, this should be set to the Flask backend URL.
 */
export const API_BASE = '';

interface APIErrorDetail {
	loc?: string[];
	msg: string;
	type?: string;
}

interface APIErrorResponse {
	error: string;
	details?: APIErrorDetail[];
}

function isAPIErrorDetail(value: unknown): value is APIErrorDetail {
	return (
		typeof value === 'object' &&
		value !== null &&
		'msg' in value &&
		typeof (value as Record<string, unknown>).msg === 'string'
	);
}

function isAPIErrorResponse(value: unknown): value is APIErrorResponse {
	return (
		typeof value === 'object' &&
		value !== null &&
		'error' in value &&
		typeof (value as Record<string, unknown>).error === 'string'
	);
}

/** Extract a human-readable error message from a failed API response.
 *
 * Technical details (status, URL, response body) are logged to the console
 * so they can be referenced during debugging.
 */
export async function apiErrorMessage(res: Response, context?: string): Promise<string> {
	let bodyText = '';
	try {
		bodyText = await res.text();
	} catch {
		/* body already consumed or unreadable */
	}

	let parsedBody: unknown;
	let errorFromBody: string | undefined;
	try {
		parsedBody = JSON.parse(bodyText);
	} catch {
		/* not JSON */
	}

	if (isAPIErrorResponse(parsedBody)) {
		if (parsedBody.details && parsedBody.details.length > 0) {
			const detailMessages = parsedBody.details
				.filter(isAPIErrorDetail)
				.map((d) => (d.loc && d.loc.length ? `${d.loc.join('.')}: ${d.msg}` : d.msg));
			if (detailMessages.length > 0) {
				errorFromBody = detailMessages.join('; ');
			}
		}
		if (!errorFromBody) {
			errorFromBody = parsedBody.error;
		}
	}

	console.error(
		`[API Error] ${context ? context + ' → ' : ''}${res.status} ${res.statusText}\n` +
			`  URL: ${res.url}\n` +
			`  Response: ${errorFromBody || bodyText || '(empty body)'}`
	);

	if (errorFromBody) {
		return errorFromBody;
	}

	return userFriendlyMessage(res.status);
}

/** Return a user-friendly message for a thrown network/fetch error.
 *
 * Logs the original error to the console for debugging.
 */
export function friendlyErrorMessage(error: unknown, fallback: string): string {
	if (error instanceof TypeError && error.message.toLowerCase().includes('fetch')) {
		console.error('[Network Error]', error);
		return "Can't connect to the server. Is it running?";
	}
	if (error instanceof Error) {
		console.error('[Network Error]', error);
		return error.message;
	}
	console.error('[Network Error]', error);
	return fallback;
}

function userFriendlyMessage(status: number): string {
	const messages: Record<number, string> = {
		400: "That request wasn't valid. Please check your input and try again.",
		401: 'You need to sign in to do that.',
		403: "You don't have permission to do that.",
		404: "That doesn't exist.",
		409: 'That conflicts with something already there.',
		422: "That input isn't valid. Please check and try again.",
		500: 'Something went wrong on the server.',
		502: "Can't connect to the backend. Is the server running?",
		503: 'The server is temporarily unavailable. Please try again later.',
		504: 'The server took too long to respond. Please try again later.'
	};
	return messages[status] || 'Something went wrong. Please try again.';
}
