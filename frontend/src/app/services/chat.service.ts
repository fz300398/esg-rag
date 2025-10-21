import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class ChatService {
  private apiUrl = environment.backendUrl;

  constructor(private http: HttpClient) {}

  // Anfrage an das Chat-Backend
  ask(question: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/query`, { question });
  }

  // Datei-Upload an das Backend
  uploadFile(formData: FormData): Observable<any> {
    return this.http.post(`${this.apiUrl}/upload`, formData);
  }
}