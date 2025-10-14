import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({ providedIn: 'root' })
export class ChatService {
  private baseUrl = 'http://localhost:8000'; // FastAPI Endpoint
  private storageKey = 'esg_chat_history';

  constructor(private http: HttpClient) {}

  getChatHistory() {
    const data = localStorage.getItem(this.storageKey);
    return data ? JSON.parse(data) : [];
  }

  saveChatHistory(history: any[]) {
    localStorage.setItem(this.storageKey, JSON.stringify(history));
  }

  async sendMessage(message: string) {
    const res = await this.http.post(`${this.baseUrl}/query`, { question: message }).toPromise();
    return res;
  }
}