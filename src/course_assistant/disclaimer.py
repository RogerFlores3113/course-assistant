"""The disclaimer (decisions doc, section 7). Shown on first run, under every AI answer, and in the README and LICENSE."""

FULL = ("Course Assistant is a study aid. It will make mistakes from time to time, so check anything important "
        "against your course materials and your instructor. It is here to help you study, not to replace your own "
        "notes, organization or judgment. The author takes no responsibility for errors in its answers or for how "
        "they are used.")

# The small footer under every AI answer.
FOOTER = "AI can make mistakes. Check important points against the cited pages and your instructor."

# Under guides an AI app (Claude Desktop, Codex) saved through the MCP server.
AI_APP_FOOTER = "Written by your AI app from your course materials. It can make mistakes; check the cited pages."

ACK_SETTING = "disclaimer_ack_at"  # settings row holding when the student pressed "I understand"
