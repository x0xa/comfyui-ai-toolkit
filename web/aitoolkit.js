import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "AIToolkit.TrainingMonitor",

    async setup() {
        // Listen for training progress updates
        api.addEventListener("aitoolkit.progress", (event) => {
            const { step, total_steps, loss } = event.detail;
            const pct = total_steps > 0 ? ((step / total_steps) * 100).toFixed(1) : 0;
            console.log(`[AI Toolkit] Step ${step}/${total_steps} (${pct}%) | Loss: ${loss.toFixed(6)}`);
        });

        // Listen for log messages
        api.addEventListener("aitoolkit.log", (event) => {
            const { message } = event.detail;
            if (message && message.trim()) {
                console.log(`[AI Toolkit] ${message}`);
            }
        });

        // Listen for new sample images
        api.addEventListener("aitoolkit.samples", (event) => {
            const { step, count, paths } = event.detail;
            console.log(`[AI Toolkit] ${count} new sample(s) at step ${step}`);
        });
    },

    async nodeCreated(node) {
        if (node.comfyClass === "AIToolkitTrainExecute") {
            // Add a status widget to the training node
            const statusWidget = node.addWidget("text", "status", "Ready", () => {}, {
                serialize: false,
            });

            const lossWidget = node.addWidget("text", "loss", "—", () => {}, {
                serialize: false,
            });

            const onProgress = (event) => {
                const { step, total_steps, loss } = event.detail;
                const pct = total_steps > 0 ? ((step / total_steps) * 100).toFixed(1) : 0;
                statusWidget.value = `Step ${step}/${total_steps} (${pct}%)`;
                lossWidget.value = `Loss: ${loss.toFixed(6)}`;
                node.setDirtyCanvas(true, true);
            };

            const onLog = (event) => {
                const { message } = event.detail;
                if (message && message.includes("Error")) {
                    statusWidget.value = `Error: ${message.slice(0, 80)}`;
                    node.setDirtyCanvas(true, true);
                }
            };

            // Register listeners when execution starts
            api.addEventListener("aitoolkit.progress", onProgress);
            api.addEventListener("aitoolkit.log", onLog);

            // Clean up on node removal
            const origOnRemoved = node.onRemoved;
            node.onRemoved = function() {
                api.removeEventListener("aitoolkit.progress", onProgress);
                api.removeEventListener("aitoolkit.log", onLog);
                origOnRemoved?.call(this);
            };
        }
    },
});
