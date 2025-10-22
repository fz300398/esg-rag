import { Component, ElementRef, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../services/chat.service';
import { AuthService } from '../services/auth.service';

interface Message {
  sender: 'user' | 'bot';
  text: string;
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css']
})
export class ChatComponent {
  @ViewChild('chatWindow', { static: true }) chatWindow!: ElementRef<HTMLDivElement>;

  sessionId = '';
  question = '';
  messages: Message[] = [];
  loading = false;

  constructor(private chat: ChatService, private auth: AuthService) {}

  ngOnInit(): void {
    this.sessionId = crypto.randomUUID();
  }

  onKeyDown(e: KeyboardEvent): void {
    if (e.key === 'Enter') {
      // Shift+Enter = neuer Zeilenumbruch
      if (e.shiftKey) return;
      // Enter = senden
      e.preventDefault();
      this.send();
    }
  }

  private pushAndScroll(msg: Message) {
    this.messages.push(msg);
    // Scroll ans Ende (kleines Timeout bis DOM gerendert ist)
    setTimeout(() => {
      if (this.chatWindow?.nativeElement) {
        const el = this.chatWindow.nativeElement;
        el.scrollTop = el.scrollHeight;
      }
    });
  }

  send(): void {
    const prompt = this.question.trim();
    if (!prompt || this.loading) return;

    // Sofort Eingabe leeren
    this.question = '';

    // User-Nachricht anzeigen
    this.pushAndScroll({ sender: 'user', text: prompt });

    // Request vorbereiten
    const payload = {
      question: prompt,
      session_id: this.sessionId
    };

    this.loading = true;
    this.chat.ask(payload).subscribe({
      next: (response: any) => {
        this.pushAndScroll({ sender: 'bot', text: response?.answer ?? 'Keine Antwort erhalten.' });
        this.loading = false;
      },
      error: (err) => {
        console.error('Fehler beim Abrufen der Antwort:', err);
        this.pushAndScroll({ sender: 'bot', text: 'Fehler beim Abrufen der Antwort.' });
        this.loading = false;
      }
    });
  }

  onFileSelected(event: any): void {
    const file = event?.target?.files?.[0];
    if (!file || this.loading) return;

    const formData = new FormData();
    formData.append('file', file);

    this.loading = true;
    this.chat.uploadFile(formData).subscribe({
      next: () => {
        this.pushAndScroll({
          sender: 'bot',
          text: `Die Datei "${file.name}" wurde erfolgreich hochgeladen.`
        });
        this.loading = false;
      },
      error: () => {
        this.pushAndScroll({
          sender: 'bot',
          text: `Fehler beim Hochladen der Datei "${file.name}".`
        });
        this.loading = false;
      }
    });
  }

  logout(): void {
    this.auth.logout();
  }
}