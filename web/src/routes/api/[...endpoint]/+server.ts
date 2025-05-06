import type { RequestHandler } from "@sveltejs/kit";
import { error } from "@sveltejs/kit";

export const GET: RequestHandler = async ({ params, cookies }) => {
    const endpoint = params.endpoint;

    if (endpoint === "") {
        return new Response();
    } else {
        throw error(404);
    }
};

export const POST: RequestHandler = async ({ params, cookies, request }) => {
    const endpoint = params.endpoint;

    if (endpoint === "ws_toast_echo") {
        const text = await request.text();
        server_session.send_ws_toast_echo(text);
        return new Response();
    } else {
        throw error(404);
    }
};
