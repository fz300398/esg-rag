import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { environment } from '../../environments/environment';
import { Observable, map } from 'rxjs';
import { AuthService } from './auth.service';

interface SourceItem {
  source: string;
  page: number;
  text?: string;
  chunk_id?: string;
}

@Injectable({ providedIn: 'root' })
export class ChatService {
  private apiUrl = environment.backendUrl;

  constructor(private http: HttpClient, private auth: AuthService) {}

  ask(payload: { question: string; session_id: string }): Observable<any> {
    const headers = this._getAuthHeaders();

    return this.http.post<any>(`${this.apiUrl}/query`, payload, { headers }).pipe(
      map((res) => {
        // === Doppelte Quellen entfernen ===
        if (Array.isArray(res.sources)) {
          const seen = new Set<string>();
          res.sources = res.sources.filter((s: SourceItem) => {
            const key = `${s.source}_${s.page}`;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
          });
        }

        // === Confidence nur anzeigen, wenn Quellen existieren ===
        if (!res.sources || res.sources.length === 0) {
          res.confidence = null;
        }

        return res;
      })
    );
  }

  uploadFiles(files: FileList): Observable<any> {
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i], files[i].name);
    }

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