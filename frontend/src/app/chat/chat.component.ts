import { Component } from '@angular/core';
import { ChatService } from '../services/chat.service';
import { AuthService } from '../services/auth.service';

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css']
})
export class ChatComponent {
  messages: { sender: string; text: string }[] = [];
  question = '';

  constructor(private chat: ChatService, private auth: AuthService) {}

  sendMessage() {
    if (!this.question.trim()) return;
    this.messages.push({ sender: 'User', text: this.question });

    this.chat.ask(this.question).subscribe({
      next: (res) => {
        this.messages.push({ sender: 'Bot', text: res.answer });
        this.question = '';
      },
      error: () => {
        this.messages.push({ sender: 'Bot', text: 'Fehler beim Laden der Antwort.' });
      }
    });
  }

  logout() {
    this.auth.logout();
    window.location.href = '/';
  }
}