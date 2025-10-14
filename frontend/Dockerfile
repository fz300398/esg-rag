FROM node:20 AS build

WORKDIR /app
COPY package*.json ./
RUN npm install -g @angular/cli@17 && npm install
COPY . .

EXPOSE 4200
CMD ["npm", "start"]
