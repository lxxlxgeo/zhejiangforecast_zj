import { createApp } from "vue";
import { VueQueryPlugin } from "@tanstack/vue-query";
import { createPinia } from "pinia";
import ElementPlus from "element-plus";
import "element-plus/dist/index.css";
import App from "./App.vue";
import { router } from "./router";
import "./styles.css";

createApp(App)
  .use(createPinia())
  .use(router)
  .use(VueQueryPlugin)
  .use(ElementPlus)
  .mount("#app");
