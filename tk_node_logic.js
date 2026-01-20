import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "TK.NodeLogic",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "TK_BatchImageLoader" || nodeData.name === "TK_JoyCaption_Interrogator") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }

                const resizeMpWidget = this.widgets.find((w) => w.name === "resize_mp");
                const resizePxWidget = this.widgets.find((w) => w.name === "resize_px");

                if (resizeMpWidget && resizePxWidget) {
                    const originalMpCallback = resizeMpWidget.callback;
                    resizeMpWidget.callback = (value) => {
                        if (value && resizePxWidget.value) {
                            resizePxWidget.value = false;
                        }
                        if (originalMpCallback) {
                            originalMpCallback(value);
                        }
                    };

                    const originalPxCallback = resizePxWidget.callback;
                    resizePxWidget.callback = (value) => {
                        if (value && resizeMpWidget.value) {
                            resizeMpWidget.value = false;
                        }
                        if (originalPxCallback) {
                            originalPxCallback(value);
                        }
                    };
                }
            };
        }
    },
});
