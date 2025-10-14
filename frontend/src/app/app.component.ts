import { Component } from '@angular/core';
import { ApiService } from './services/api.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  question = '';
  answer = '';
  contexts: any[] = [];
  uploading = false;

  constructor(private api: ApiService) {}

  ask() {
    if (!this.question.trim()) return;
    this.answer = 'Anfrage wird verarbeitet...';
    this.api.query(this.question).subscribe({
      next: (res) => {
        this.answer = res.answer;
        this.contexts = res.contexts || [];
      },
      error: () => (this.answer = 'Fehler bei der Anfrage.')
    });
  }

  uploadFile(event: any) {
    const file = event.target.files[0];
    if (!file) return;
    this.uploading = true;
    this.api.upload(file).subscribe({
      next: (res) => {
        this.uploading = false;
        alert(res.msg || 'Upload erfolgreich!');
      },
      error: () => {
        this.uploading = false;
        alert('Upload fehlgeschlagen.');
      }
    });
  }
}