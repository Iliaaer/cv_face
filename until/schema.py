from pydantic import BaseModel
import datetime


class GroupStudents(BaseModel):
    number: str


class Student(BaseModel):
    name: str
    last_name: str
    middle_name: str
    group_id: int


class Frame(BaseModel):
    name: str


class Audience(BaseModel):
    # frame: Frame
    audience: int
    frame_id: int


class Discipline(BaseModel):
    name: str
    audience_id: int
    group_id: int
    time: datetime.time
    data: datetime.date


class Attendance(BaseModel):
    student_id: int
    discipline_id: int
    time: datetime.time
