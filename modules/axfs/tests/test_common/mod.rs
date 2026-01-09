use axfs::api as fs;
use axio as io;

use fs::{File, FileType, OpenOptions};
use io::{Error, Result, prelude::*};

macro_rules! assert_err {
    ($expr: expr) => {
        assert!(($expr).is_err())
    };
    ($expr: expr, $err: ident) => {
        assert_eq!(($expr).err(), Some(Error::$err))
    };
}

fn test_read_write_file() -> Result<()> {
    let fname = "///very/long//.././long//./path/./test.txt";
    println!("read and write file {fname:?}:");

    // read and write
    let mut file = File::options().read(true).write(true).open(fname)?;
    let file_size = file.metadata()?.len();
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    print!("{contents}");
    assert_eq!(contents.len(), file_size as usize);
    assert_eq!(file.write(b"Hello, world!\n")?, 14); // append
    drop(file);

    // read again and check
    let new_contents = fs::read_to_string(fname)?;
    print!("{new_contents}");
    assert_eq!(new_contents, contents + "Hello, world!\n");

    // append and check
    let mut file = OpenOptions::new().append(true).open(fname)?;
    assert_eq!(file.write(b"new line\n")?, 9);
    drop(file);

    let new_contents2 = fs::read_to_string(fname)?;
    print!("{new_contents2}");
    assert_eq!(new_contents2, new_contents + "new line\n");

    // open a non-exist file
    assert_err!(File::open("/not/exist/file"), NotFound);

    println!("test_read_write_file() OK!");
    Ok(())
}

fn test_read_dir() -> Result<()> {
    let dir = "/././//./";
    println!("list directory {dir:?}:");
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        println!("   {}", entry.file_name());
    }
    println!("test_read_dir() OK!");
    Ok(())
}

fn test_file_permission() -> Result<()> {
    let fname = "./short.txt";
    println!("test permission {fname:?}:");

    // write a file that open with read-only mode
    let mut buf = [0; 256];
    let mut file = File::open(fname)?;
    let n = file.read(&mut buf)?;
    assert_err!(file.write(&buf), PermissionDenied);
    drop(file);

    // read a file that open with write-only mode
    let mut file = File::create(fname)?;
    assert_err!(file.read(&mut buf), PermissionDenied);
    assert!(file.write(&buf[..n]).is_ok());
    drop(file);

    // open with empty options
    assert_err!(OpenOptions::new().open(fname), InvalidInput);

    // read as a directory
    assert_err!(fs::read_dir(fname), NotADirectory);
    assert_err!(fs::read("short.txt/"), NotADirectory);
    assert_err!(fs::metadata("/short.txt/"), NotADirectory);

    // create as a directory
    assert_err!(fs::write("error/", "should not create"), NotADirectory);
    assert_err!(fs::metadata("error/"), NotFound);
    assert_err!(fs::metadata("error"), NotFound);

    // read/write a directory
    assert_err!(fs::write(".", "test"), IsADirectory);

    println!("test_file_permisson() OK!");
    Ok(())
}

fn test_create_file_dir() -> Result<()> {
    // create a file and test existence
    let fname = "././/very-long-dir-name/..///new-file.txt";
    println!("test create file {fname:?}:");
    assert_err!(fs::metadata(fname), NotFound);
    let contents = "create a new file!\n";
    fs::write(fname, contents)?;

    let dirents = fs::read_dir(".")?
        .map(|e| e.unwrap().file_name())
        .collect::<Vec<_>>();
    println!("dirents = {dirents:?}");
    assert!(dirents.contains(&"new-file.txt".into()));
    assert_eq!(fs::read_to_string(fname)?, contents);
    assert_err!(File::create_new(fname), AlreadyExists);

    // create a directory and test existence
    let dirname = "///././/very//.//long/./new-dir";
    println!("test create dir {dirname:?}:");
    assert_err!(fs::metadata(dirname), NotFound);
    fs::create_dir(dirname)?;

    let dirents = fs::read_dir("./very/long")?
        .map(|e| e.unwrap().file_name())
        .collect::<Vec<_>>();
    println!("dirents = {dirents:?}");
    assert!(dirents.contains(&"new-dir".into()));
    assert!(fs::metadata(dirname)?.is_dir());
    assert_err!(fs::create_dir(dirname), AlreadyExists);

    println!("test_create_file_dir() OK!");
    Ok(())
}

fn test_remove_file_dir() -> Result<()> {
    // remove a file and test existence
    let fname = "//very-long-dir-name/..///new-file.txt";
    println!("test remove file {fname:?}:");
    assert_err!(fs::remove_dir(fname), NotADirectory);
    assert!(fs::remove_file(fname).is_ok());
    assert_err!(fs::metadata(fname), NotFound);
    assert_err!(fs::remove_file(fname), NotFound);

    // remove a directory and test existence
    let dirname = "very//.//long/../long/.//./new-dir////";
    println!("test remove dir {dirname:?}:");
    assert_err!(fs::remove_file(dirname), IsADirectory);
    assert!(fs::remove_dir(dirname).is_ok());
    assert_err!(fs::metadata(dirname), NotFound);
    assert_err!(fs::remove_dir(fname), NotFound);

    // error cases
    assert_err!(fs::remove_file(""), NotFound);
    assert_err!(fs::remove_dir("/"), DirectoryNotEmpty);
    assert_err!(fs::remove_dir("."), InvalidInput);
    assert_err!(fs::remove_dir("../"), InvalidInput);
    assert_err!(fs::remove_dir("./././/"), InvalidInput);
    assert_err!(fs::remove_file("///very/./"), IsADirectory);
    assert_err!(fs::remove_file("short.txt/"), NotADirectory);
    assert_err!(fs::remove_dir(".///"), InvalidInput);
    assert_err!(fs::remove_dir("/./very///"), DirectoryNotEmpty);
    assert_err!(fs::remove_dir("very/long/.."), InvalidInput);

    println!("test_remove_file_dir() OK!");
    Ok(())
}

pub fn test_all() {
    test_read_write_file().expect("test_read_write_file() failed");
    test_read_dir().expect("test_read_dir() failed");
    test_file_permission().expect("test_file_permission() failed");
    test_create_file_dir().expect("test_create_file_dir() failed");
    test_remove_file_dir().expect("test_remove_file_dir() failed");
}
