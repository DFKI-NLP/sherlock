"""
python -m sherlock.microscope.server \
    --pipeline-config-path <DIRECTORY>/ \
    --title "Sherlock Demo"
```
"""
import argparse
import importlib
import logging
import os
import pkgutil
import sys
from pathlib import Path
from typing import Optional

from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from sherlock import Document
from sherlock.microscope.conversion import document_to_brat
from sherlock.pipeline import Pipeline


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ServerError(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        error_dict = dict(self.payload or ())
        error_dict["message"] = self.message
        return error_dict


def make_app(
    pipeline: Pipeline,
    examples_file: str,
    static_dir: Optional[str] = None,
    title: str = "Sherlock Demo",
) -> Flask:
    """
    Creates a Flask app that serves up the provided pipeline
    along with a front-end for interacting with it.
    If you would rather create your own HTML, call it index.html
    and provide its directory as ``static_dir``.
    """
    if static_dir is not None:
        static_dir = os.path.abspath(static_dir)
        if not os.path.exists(static_dir):
            logger.error("app directory %s does not exist, aborting", static_dir)
            sys.exit(-1)

    app = Flask(__name__)  # pylint: disable=invalid-name

    @app.errorhandler(ServerError)
    def handle_invalid_usage(error: ServerError) -> Response:  # pylint: disable=unused-variable
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @app.route("/")
    def index() -> Response:  # pylint: disable=unused-variable
        if static_dir is not None:
            return send_from_directory(static_dir, "index.html")
        else:
            raise ServerError("static_dir not specified", 404)

    @app.route("/examples")
    def get_examples() -> Response:  # pylint: disable=unused-variable
        return send_file(examples_file)

    @app.route("/predict", methods=["POST", "OPTIONS"])
    def predict() -> Response:  # pylint: disable=unused-variable
        """make a prediction using the specified pipeline and return the results"""
        if request.method == "OPTIONS":
            return Response(response="", status=200)

        output_format = request.args.get("format", "json").lower()
        if output_format not in ["json", "brat", "json-brat"]:
            raise ServerError(f"Unknown output format: {output_format}")

        data = request.get_json()
        logger.info("data: %s", data)

        doc = Document(data.get("guid", 0), data["text"])
        pipeline.predict_document(doc)

        logger.info("document: %s", doc)

        if output_format == "json":
            result_data = doc.to_dict()
        elif output_format == "brat":
            result_data = document_to_brat(doc)
        elif output_format == "json-brat":
            result_data = document_to_brat(doc)
            result_data["json"] = doc.to_dict()
        else:
            raise Exception("This should not happen.")

        logger.info("result: %s", result_data)
        return jsonify(result_data)

    # @app.route('/predict_batch', methods=['POST', 'OPTIONS'])
    # def predict_batch() -> Response:  # pylint: disable=unused-variable
    #     """make a prediction using the specified model and return the results"""
    #     if request.method == "OPTIONS":
    #         return Response(response="", status=200)

    #     return jsonify(data)

    @app.route("/<path:path>")
    def static_proxy(path: str) -> Response:  # pylint: disable=unused-variable
        if static_dir is not None:
            return send_from_directory(static_dir, path)
        else:
            raise ServerError("static_dir not specified", 404)

    return app


def _import_submodules(package_name: str) -> None:
    """
    Import all submodules under the given package.
    Primarily useful to have custom classes get loaded and registered.
    """
    # Code taken from https://github.com/allenai/allennlp/blob/v0.9.0/allennlp/common/util.py#L308
    importlib.invalidate_caches()

    # For some reason, python doesn't always add this by default to your path, but you pretty much
    # always want it when using `--include-package`.  And if it's already there, adding it again at
    # the end won't hurt anything.
    sys.path.append(".")

    # Import at top level
    module = importlib.import_module(package_name)
    path = getattr(module, "__path__", [])
    path_string = "" if not path else path[0]

    # walk_packages only finds immediate children, so need to recurse.
    for module_finder, name, _ in pkgutil.walk_packages(path):
        # Sometimes when you import third-party libraries that are on your path,
        # `pkgutil.walk_packages` returns those too, so we need to skip them.
        if path_string and module_finder.path != path_string:
            continue
        subpackage = f"{package_name}.{name}"
        _import_submodules(subpackage)


def main(args):
    parser = argparse.ArgumentParser(description="Serve up a simple model")

    parser.add_argument(
        "--pipeline-config", type=str, required=True, help="path to the pipeline configuration file"
    )
    parser.add_argument("--cuda-device", type=int, default=-1, help="id of GPU to use (if any)")
    parser.add_argument(
        "--include-package",
        type=str,
        action="append",
        default=[],
        help="additional packages to include",
    )
    parser.add_argument(
        "-o",
        "--overrides",
        type=str,
        default="",
        help="a JSON structure used to override the pipeline configuration",
    )
    parser.add_argument(
        "--static-dir", type=str, default=None, help="serve index.html from this directory"
    )
    parser.add_argument(
        "--title", type=str, help="change the default page title", default="Sherlock Demo"
    )
    parser.add_argument("--port", type=int, default=8000, help="port to serve the demo on")
    parser.add_argument("--rest-only", action="store_true")
    parser.add_argument(
        "--examples-file", type=str, default=None, help="change the default example texts",
    )

    args = parser.parse_args(args)

    # Import any additional modules needed (to register custom classes)
    for package_name in getattr(args, "include_package", ()):
        _import_submodules(package_name)

    pipeline = Pipeline.from_file(args.pipeline_config, args.cuda_device)

    static_dir = None
    if not args.rest_only:
        if args.static_dir is None:
            static_dir = os.path.dirname(os.path.realpath(__file__)) + "/static/"
        else:
            static_dir = args.static_dir

    if args.examples_file is None:
        examples_file = "static/js/example-texts.js"
    else:
        examples_file = str(Path(args.examples_file).resolve())

    app = make_app(
        pipeline=pipeline, examples_file=examples_file, static_dir=static_dir, title=args.title,
    )
    CORS(app)

    http_server = WSGIServer(("0.0.0.0", args.port), app)
    print(f"Model loaded, serving demo on port {args.port}")
    http_server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
