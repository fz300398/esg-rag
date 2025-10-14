import { Component } from '@angular/core';
import { Router } from '@angular/router';
import { AuthService } from '../services/auth.service';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent {
  constructor(private auth: AuthService, private router: Router) {}

  login() {
    // Simulierter THWS-Login: normalerweise Redirect zur Auth-URL
    const fakeToken = 'thws-demo-token';
    this.auth.saveToken(fakeToken);
    this.router.navigate(['/chat']);
  }
}