import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { environment } from '../../environments/environment';
import { Observable } from 'rxjs';
import { AuthService } from './auth.service';

@Injectable({ providedIn: 'root' })
export class ChatService {
  private apiUrl = environment.backendUrl;

  constructor(private http: HttpClient, private auth: AuthService) {}

  ask(payload: { question: string; session_id: string }): Observable<any> {
      const headers = this._getAuthHeaders();
      return this.http.post(`${this.apiUrl}/query`, payload, { headers });
    }

    uploadFile(formData: FormData): Observable<any> {
      const headers = this._getAuthHeaders();
      return this.http.post(`${this.apiUrl}/upload`, formData, { headers });
    }

    private _getAuthHeaders(): HttpHeaders {
      const token = this.auth.getToken();
      let headers = new HttpHeaders();
      if (token) {
        headers = headers.set('Authorization', `Bearer ${token}`);
      }
      return headers;
    }
}