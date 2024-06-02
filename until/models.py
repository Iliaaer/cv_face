from sqlalchemy import Column, Integer, String, ForeignKey, Date, Time
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class GroupStudents(Base):
    __tablename__ = 'group_students'
    id = Column(Integer, primary_key=True, index=True)
    number = Column(String, )


class Student(Base):
    __tablename__ = 'student'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    last_name = Column(String)
    middle_name = Column(String)
    group_id = Column(Integer, ForeignKey('group_students.id'))
    group = relationship('GroupStudents')


class Frame(Base):
    __tablename__ = 'frame'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)


class Audience(Base):
    __tablename__ = 'audience'
    id = Column(Integer, primary_key=True, index=True)
    audience = Column(Integer)
    frame_id = Column(Integer, ForeignKey('frame.id'))
    frame = relationship('Frame')


class Discipline(Base):
    __tablename__ = 'discipline'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    audience_id = Column(Integer, ForeignKey("audience.id"))
    audience = relationship('Audience')
    group_id = Column(Integer, ForeignKey('group_students.id'))
    group = relationship('GroupStudents')
    time = Column(Time)
    data = Column(Date)


class Attendance(Base):
    __tablename__ = 'attendance'
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey('student.id'))
    student = relationship('Student')
    discipline_id = Column(Integer, ForeignKey('discipline.id'))
    discipline = relationship('Discipline')
    time = Column(Time)
    # data = Column(Date)


class PathStudetnsFiles(Base):
    __tablename__ = 'path to studetns files'
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey('student.id'))
    student = relationship('Student')
    path = Column(String)
