import type { RequestHandler } from "@sveltejs/kit";
import { error } from "@sveltejs/kit";
import axios from "axios";
import { SHADERBOX_URL } from "$env/static/private";

export const GET: RequestHandler = async ({ params, cookies }) => {
    const endpoint = params.endpoint;

    if (endpoint === "") {
        return new Response();
    } else {
        throw error(404, "Endpoint not found");
    }
};

export const POST: RequestHandler = async ({ params, cookies, request }) => {
    const endpoint = params.endpoint;

    if (endpoint === "apply_node") {
        try {
            const form_data = await request.formData();

            const response = await axios.post(
                `${SHADERBOX_URL}/apply_node`,
                form_data,
                {
                    headers: {
                        "Content-Type": "multipart/form-data"
                    }
                }
            );

            return new Response(JSON.stringify(response.data), {
                status: response.status,
                headers: {
                    "Content-Type": "application/json"
                }
            });
        } catch (err) {
            console.error(err);

            if (axios.isAxiosError(err) && err.response) {
                throw error(
                    err.response.status,
                    err.response.data?.detail || "Backend error"
                );
            }
            throw error(500, "Internal server error");
        }
    } else {
        throw error(404, "Endpoint not found");
    }
};
