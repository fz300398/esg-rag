import { Component, ElementRef, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../services/chat.service';
import { AuthService } from '../services/auth.service';

interface Message {
  sender: 'user' | 'bot';
  text: string;
  sources?: { source: string; page: number }[]; // Quellenanzeige
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
  selectedFiles: FileList | null = null;

  constructor(private chat: ChatService, private auth: AuthService) {}

  ngOnInit(): void {
    this.sessionId = crypto.randomUUID();
  }

  onKeyDown(e: KeyboardEvent): void {
    if (e.key === 'Enter') {
      if (e.shiftKey) return;
      e.preventDefault();
      this.send();
    }
  }

  private pushAndScroll(msg: Message) {
    this.messages.push(msg);
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
    this.question = '';
    this.pushAndScroll({ sender: 'user', text: prompt });

    const payload = {
      question: prompt,
      session_id: this.sessionId
    };

    this.loading = true;
    this.chat.ask(payload).subscribe({
      next: (response: any) => {
        // Quellen aus Backend übernehmen
        this.pushAndScroll({
          sender: 'bot',
          text: response?.answer ?? 'Keine Antwort erhalten.',
          sources: response?.sources ?? []
        });
        this.loading = false;
      },
      error: (err) => {
        console.error('Fehler beim Abrufen der Antwort:', err);
        this.pushAndScroll({ sender: 'bot', text: 'Fehler beim Abrufen der Antwort.' });
        this.loading = false;
      }
    });
  }

  onFilesSelected(event: any): void {
    this.selectedFiles = event?.target?.files ?? null;
  }

  uploadFiles(): void {
    if (!this.selectedFiles || this.loading) return;

    this.loading = true;
    this.chat.uploadFiles(this.selectedFiles).subscribe({
      next: (res) => {
        const count = res?.files?.length ?? this.selectedFiles?.length;
        this.pushAndScroll({
          sender: 'bot',
          text: `${count} Datei(en) erfolgreich hochgeladen und indexiert.`
        });
        this.loading = false;
        this.selectedFiles = null;
      },
      error: (err) => {
        console.error('Upload-Fehler:', err);
        this.pushAndScroll({ sender: 'bot', text: 'Fehler beim Upload der Dateien.' });
        this.loading = false;
      }
    });
  }

  logout(): void {
    this.auth.logout();
  }
}