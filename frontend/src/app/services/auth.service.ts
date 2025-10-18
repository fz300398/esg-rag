import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Router } from '@angular/router';
import { firstValueFrom } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class AuthService {
  private loginUrl = 'https://api.fiw.fhws.de/auth/api/users/me';

  constructor(private http: HttpClient, private router: Router) {}

  async login(username: string, password: string): Promise<boolean> {
    const headers = new HttpHeaders({
      Authorization: 'Basic ' + btoa(`${username}:${password}`),
    });

    try {
      const observable$ = this.http.get(this.loginUrl, {
        headers,
        observe: 'response',
      });

      const response = await firstValueFrom(observable$);

      if (response.status === 200) {
        const token = response.headers.get('X-fhws-jwt-token');
        if (token) {
          localStorage.setItem('token', token);
          localStorage.setItem('username', username);
          return true;
        }
      }

      return false;
    } catch (error) {
      console.error('Login fehlgeschlagen:', error);
      return false;
    }
  }

  logout(): void {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    this.router.navigate(['/login']);
  }

  isLoggedIn(): boolean {
    return !!localStorage.getItem('token');
  }

  getToken(): string | null {
    return localStorage.getItem('token');
  }
}