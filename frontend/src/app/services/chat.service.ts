import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '@env/environment';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class ChatService {
  private apiUrl = `${environment.backendUrl}/query`;

  constructor(private http: HttpClient) {}

  ask(question: string): Observable<any> {
    return this.http.post(this.apiUrl, { question });
  }
}