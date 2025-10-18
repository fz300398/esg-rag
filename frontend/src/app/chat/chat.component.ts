import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../services/chat.service';
import { AuthService } from '../services/auth.service';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css']
})
export class ChatComponent {
  question = '';
  messages: { sender: string; text: string }[] = [];

  constructor(private chat: ChatService, private auth: AuthService) {}

  send() {
    if (!this.question.trim()) return;

    const userMessage = this.question;
    this.messages.push({ sender: 'user', text: userMessage });
    this.question = '';

    this.chat.ask(userMessage).subscribe({
      next: (response: any) => {
        this.messages.push({ sender: 'bot', text: response.answer });
      },
      error: () => {
        this.messages.push({ sender: 'bot', text: 'Fehler beim Abrufen der Antwort.' });
      }
    });
  }

  logout() {
    this.auth.logout();
  }
}