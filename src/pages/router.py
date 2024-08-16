from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter(
    prefix='',
    tags=['Pages']
)

templates = Jinja2Templates(directory='templates')


@router.get('/')
def get_main_page(request: Request):
    return templates.TemplateResponse('base.html', {'request': request})


@router.get('/predict')
def get_predict_page(request: Request):
    return templates.TemplateResponse('base.html', {'request': request})