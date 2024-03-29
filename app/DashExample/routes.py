from . import blueprint
from flask import render_template
from flask_login import login_required
from Dashboard import Dash_App1, Dash_App2, Dash_App3, Dash_App4

@blueprint.route('/app1')
@login_required
def app1_template():
    return render_template('app1.html', dash_url = Dash_App1.url_base)

@blueprint.route('/app2')
@login_required
def app2_template():
    return render_template('app2.html', dash_url = Dash_App2.url_base)

@blueprint.route('/app3')
@login_required
def app3_template():
    return render_template('app3.html', dash_url = Dash_App3.url_base)

@blueprint.route('/app4')
@login_required
def app4_template():
    return render_template('app4.html', dash_url = Dash_App4.url_base)