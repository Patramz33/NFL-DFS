# NFL DFS Dashboard V2.1 — iPhone / Cloud Deployment

## Recommended: Streamlit Community Cloud

You need a GitHub account and a Streamlit Community Cloud account.

1. Create a new GitHub repository, for example `nfl-dfs-dashboard`.
2. Upload the contents of this folder to the repository root. Keep `.streamlit/config.toml` in its `.streamlit` folder.
3. Go to https://share.streamlit.io and sign in with GitHub.
4. Create a new app and select your repository.
5. Set the entrypoint file to `app.py`.
6. Pick a Python version compatible with the packages in `requirements.txt` (Python 3.12 is a conservative choice).
7. Deploy.
8. Open the generated Streamlit app URL on your iPhone in Safari.
9. Tap Safari's Share button, choose **Add to Home Screen**, edit the name if desired, and tap Add.

## Weekly workflow on iPhone

1. Tap the dashboard icon from your Home Screen.
2. Open the sidebar and upload your prepared DFS player-pool CSV.
3. Choose the Dashboard view from the selector near the top.
4. Set optimizer/portfolio controls in the sidebar.
5. Generate lineups and download lineup/portfolio CSV files to the iPhone Files app.

## Notes

- The included Week 1 player pool is synthetic test data, not live DraftKings salaries or projections.
- Community Cloud redeploys when you push changes to the connected GitHub repository.
- Do not commit private API keys or credentials. If V3 later adds authenticated data feeds, use Streamlit Secrets instead.
