import { defineStore } from "pinia";

const STORAGE_KEY = "zj_online_modeling_recent_tasks";

export const useTaskStore = defineStore("taskStore", {
  state: () => ({
    activeTaskId: "",
    activeJobId: "",
    recentTasks: loadRecentTasks()
  }),
  actions: {
    setActiveTask(taskId: string) {
      this.activeTaskId = taskId;
      if (taskId) {
        this.recentTasks = [taskId, ...this.recentTasks.filter((item) => item !== taskId)].slice(0, 12);
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(this.recentTasks));
      }
    },
    setActiveJob(jobId?: string | null) {
      this.activeJobId = jobId ?? "";
    }
  }
});

function loadRecentTasks() {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    return value ? (JSON.parse(value) as string[]) : [];
  } catch {
    return [];
  }
}
