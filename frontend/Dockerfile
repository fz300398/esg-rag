FROM node:20-slim

WORKDIR /app

# Package-Dateien kopieren
COPY package*.json ./

# Angular CLI + Dependencies installieren (keine optionalen Rollup-Pakete!)
RUN npm install -g @angular/cli@17 \
    && npm install --omit=optional --legacy-peer-deps

# Quellcode kopieren
COPY . .

# Rollup vollständig entfernen (verhindert den Bug)
RUN npm uninstall rollup --force || true

EXPOSE 4200

# Webpack anstelle von Rollup aktivieren
CMD ["npx", "ng", "serve", "--no-prebundle", "--disable-host-check", "--host", "0.0.0.0", "--port", "4200"]