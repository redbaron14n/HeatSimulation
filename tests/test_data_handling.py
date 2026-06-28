from services.data_handling import DataHandler
from pathlib import Path
from pytest import raises


def test_no_extension():

    data = DataHandler("test", load=False)
    assert data.filepath == Path("Data/test.hdf5")


def test_bad_extension():

    with raises(ValueError):
        DataHandler("test.csv", load=False)
    

def test_bad_filename():

    with raises(ValueError):
        DataHandler("t*st.hdf5", load=False)


def test_double_extension():

    with raises(ValueError):
        DataHandler("test.hdf5.csv", load=False)


def test_good_filename():

    data = DataHandler("test.hdf5", load=False)
    assert data.filepath == Path("Data/test.hdf5")