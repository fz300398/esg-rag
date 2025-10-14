import { Component, OnInit } from '@angular/core';
import { ChatService } from '../../services/chat.service';

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css']
})
export class ChatComponent implements OnInit {
  message = '';
  chatHistory: { user: string; bot: string }[] = [];

  constructor(private chatService: ChatService) {}

  ngOnInit() {
    this.chatHistory = this.chatService.getChatHistory();
  }

  async sendMessage() {
    if (!this.message.trim()) return;

    const userMessage = this.message;
    this.chatHistory.push({ user: userMessage, bot: '...' });
    this.message = '';

    const response: any = await this.chatService.sendMessage(userMessage);
    const botMessage = response?.answer || '[Fehler bei der Anfrage]';

    this.chatHistory[this.chatHistory.length - 1].bot = botMessage;
    this.chatService.saveChatHistory(this.chatHistory);
  }
}