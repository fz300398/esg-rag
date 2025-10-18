import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { AuthService } from '../services/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css'],
})
export class LoginComponent {
  username = '';
  password = '';
  loading = false;
  error = '';

  constructor(private auth: AuthService, private router: Router) {}

  async login() {
    this.error = '';
    this.loading = true;

    try {
      const success = await this.auth.login(this.username, this.password);

      if (success) {
        console.log('Login erfolgreich');
        this.router.navigate(['/chat']);
      } else {
        this.error = 'Login fehlgeschlagen. Bitte überprüfe Benutzername und Passwort.';
      }
    } catch (err) {
      console.error('Login-Fehler:', err);
      this.error = 'Fehler beim Login.';
    } finally {
      this.loading = false;
    }
  }
}