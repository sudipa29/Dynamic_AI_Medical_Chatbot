from flask import Flask, render_template
from config import Config

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Application configuration
    app.config["APP_NAME"] = Config.APP_NAME
    app.config["APP_VERSION"] = Config.APP_VERSION
    app.config["MAX_MESSAGE_LENGTH"] = Config.MAX_MESSAGE_LENGTH

    @app.route("/")
    def home():
        """Render the main chatbot page."""
        return render_template(
            "index.html",
            app_name=Config.APP_NAME,
            app_version=Config.APP_VERSION,
        )

    return app

app = create_app()

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=Config.FLASK_DEBUG,
    )