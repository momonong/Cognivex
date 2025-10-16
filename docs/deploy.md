# Deployment Guide for fmri-app

## Deployment Summary

**Notice:**  
In our environment, running via a pm2 ecosystem.config.js file with environment variables did not work as expected for Poetry+Streamlit apps.  
The most reliable way is to export the required environment variables in your terminal, then use pm2 to start the app with the command line.

---

## Step-by-step

1. **Export the required environment variables before launch:**

    ```
    export GOOGLE_GENAI_USE_VERTEXAI=TRUE
    ```

2. **Start the app using pm2 from the command line:**

    ```
    pm2 start "poetry run streamlit run app.py --server.port 8501" --name "fmri-app"
    ```

3. **Check pm2 process list and logs:**

    ```
    pm2 ls
    pm2 logs fmri-app
    ```

---

## Remarks

- Using `ecosystem.config.js` with `script: "poetry", args: ...` or `script: "poetry run streamlit run ..."` may result in command not found or arguments not parsed correctly.
- Direct command line launching after exporting variables always works.
- Always verify that Poetry and Streamlit are available in your PATH for the pm2 user/session.

---

## Troubleshooting

- If you see errors like `poetry run: command not found` or `Missing key inputs argument!`, re-check your exports and try re-launching from a new shell.
- For persistent deployment with auto-restart, always ensure you manually export env variables before running pm2.

