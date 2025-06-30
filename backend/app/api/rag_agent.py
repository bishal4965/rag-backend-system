from fastapi import FastAPI, APIRouter


router = APIRouter()

@router.get('/')
def get_file():
    return 'Hi'

