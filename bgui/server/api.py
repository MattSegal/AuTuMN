import os
import sys
import logging
from functools import wraps
import traceback

from flask import Flask, abort, jsonify, current_app, request, helpers, json, make_response
from flask.ext.sqlalchemy import SQLAlchemy
import redis
from flask.ext.login import LoginManager, current_user
from werkzeug.utils import secure_filename

# Create flask app

app = Flask(__name__)

# Load configuration

app.config.from_pyfile('config.py')

# Setup logger

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s '
    '[in %(pathname)s:%(lineno)d]'
))
app.logger.addHandler(stream_handler)
app.logger.setLevel(logging.DEBUG)

# Setup databases and save in module for access

import server.dbconn
server.dbconn.redis = redis.StrictRedis.from_url(app.config["REDIS_URL"])
server.dbconn.db = SQLAlchemy(app)

# Initialize postgres database

server.dbconn.db.engine.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
server.dbconn.db.create_all()

# Cleanup dangling worklogs for parallel processing

from server import dbmodel
dbmodel.cleanup_tasklogs(server.dbconn.db.session)

# Setup login manager

login_manager = LoginManager()
login_manager.init_app(app)

# Define the user model used by flask_login

@login_manager.user_loader
def load_user(user_id):
    try:
        user = dbmodel.load_user(id=user_id)
    except Exception:
        user = None
    return user

@login_manager.unauthorized_handler
def unauthorized_handler():
    abort(401)

# Setup RPC handlers

def report_exception_decorator(api_call):
    @wraps(api_call)
    def _report_exception(*args, **kwargs):
        from werkzeug.exceptions import HTTPException
        try:
            return api_call(*args, **kwargs)
        except Exception as e:
            exception = traceback.format_exc()
            # limiting the exception information to 10000 characters maximum
            # (to prevent monstrous sqlalchemy outputs)
            current_app.logger.error("Exception during request %s: %.10000s" % (request, exception))
            if isinstance(e, HTTPException):
                raise
            code = 500
            reply = {'exception': exception}
            return make_response(jsonify(reply), code)

    return _report_exception

def get_post_data_json():
    return json.loads(request.data)

def run_fn(fn_name, args, kwargs):
    if fn_name.startswith("admin_"):
        if (current_user.is_anonymous()) \
                or not current_user.is_authenticated() \
                or not current_user.is_admin:
            return app.login_manager.unauthorized()
    elif fn_name.startswith("login_"):
        if not current_user.is_authenticated():
            return app.login_manager.unauthorized()
    elif not fn_name.startswith("public_"):
      raise ValueError('Function "%s" not valid' % (fn_name))

    # deferred importing to avoid circular imports when
    # tasks tries to import app from api.py
    if 'handler' not in globals():
        global handler
        import handler

    if hasattr(handler, fn_name):
        fn = getattr(handler, fn_name)
        print('>> RPC.handler.%s args=%s kwargs=%s' % (fn_name, args, kwargs))
    else:
        print('>> Function "%s" does not exist' % (fn_name))
        raise ValueError('Function "%s" does not exist' % (fn_name))

    return fn(*args, **kwargs)

# NOTE: the twisted wgsi server is set up to
# only allows url's with /api/* to be served

@app.route('/api', methods=['GET'])
def root():
    """ API root, nothing interesting here """
    return 'rpc-json'


@app.route('/api/rpc-run', methods=['POST'])
@report_exception_decorator
def run_remote_procedure():
    """
    post-data:
        'name': string name of function in handler
        'args': list of arguments for the function
        'kwargs: dictionary of keyword arguments
    """
    json = get_post_data_json()
    fn_name = json['name']
    args = json.get('args', [])
    kwargs = json.get('kwargs', {})

    result = run_fn(fn_name, args, kwargs)

    if result is None:
        return ''
    else:
        return jsonify(result)

@app.route('/api/rpc-download', methods=['POST'])
@report_exception_decorator
def send_downloadable_file():
    """
    url-args:
        'procedure': string name of function in handler
        'args': list of arguments for the function
    """
    post_data = get_post_data_json()

    fn_name = post_data['name']

    args = post_data.get('args', [])
    kwargs = post_data.get('kwargs', {})

    result = run_fn(fn_name, args, kwargs)

    response = helpers.send_from_directory(
        result["dirname"],
        result["basename"],
        as_attachment=True,
        attachment_filename=result["basename"])

    response.status_code = 201

    if "data" in result:
        response.headers["data"] = json.dumps(result["data"])
    response.headers["filename"] = result["basename"]

    return response

@app.route('/api/rpc-upload', methods=['POST'])
@report_exception_decorator
def receive_uploaded_file():
    """
    file-upload
    request-form:
        name: name of project
        args: string of JSON.stringify object
    """
    file = request.files['file']
    filename = secure_filename(file.filename)
    dirname = current_app.config['SAVE_FOLDER']
    if not (os.path.exists(dirname)):
        os.makedirs(dirname)
    uploaded_fname = os.path.join(dirname, filename)
    file.save(uploaded_fname)
    print("> Saved uploaded file '%s'" % (uploaded_fname))

    fn_name = request.form.get('name')
    args = json.loads(request.form.get('args', "[]"))
    kwargs = json.loads(request.form.get('kwargs', "{}"))
    args.insert(0, uploaded_fname)

    result = run_fn(fn_name, args, kwargs)

    if result is None:
        return ''
    else:
        return jsonify(result)

# Allow Cross-Origin-Resource-Sharing, mainly for working with hot reloading webclient
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:8080')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Main loop

if __name__ == '__main__':
    app.run(threaded=True, debug=True, use_debugger=False)