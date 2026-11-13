# Privacy

Course Assistant keeps your data on your computer.

- **Where it lives.** One SQLite database and a folder of your original files, in the app's data folder (see the
  README for the path on your system). Nothing is uploaded to a server run by this project, and there is no account.
- **No telemetry.** The app does not send usage data, crash reports or analytics anywhere.
- **When it uses the network.** Only when you ask: fetching pages you chose for past offerings (it follows robots.txt,
  goes slowly, sends no cookies and never signs in anywhere), and sending your question plus the matching passages from
  your materials to the AI service you set up. Transcription of lecture recordings happens on your computer; the first time you use it, the speech model is downloaded once from the internet.
- **Your AI provider.** In-app chat uses your own Anthropic API key or a local model server. If you use Anthropic,
  your questions and the passages sent with them are handled under Anthropic's terms. With a local model server
  nothing leaves your computer. If you connect Claude Desktop or Codex, what they see is governed by those apps.
- **Your key.** An API key you enter is stored in the local database and used only to call your chosen provider.
- **Local only.** The app listens on 127.0.0.1 only, locked with a per-install key, so other programs and accounts on
  your computer cannot open it.
- **Deleting.** Delete a course in the app to remove its materials, or delete the data folder to remove everything.
  The AI apps connected through MCP can add and read, never delete.
- **Your materials.** You are responsible for saving only materials you are allowed to keep, and for following each
  site's terms of use for anything you fetch.
