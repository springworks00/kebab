#![feature(generators)]
#![feature(iter_from_generator)]

use std::time::{Duration, Instant};
use std::collections::VecDeque;
use anyhow::{Result, bail, ensure, anyhow};
use std::cell::UnsafeCell;

use opencv::features2d::BRISK;
use opencv::core::{Mat, Vector, KeyPoint, Ptr, DMatch};
use opencv::prelude::{MatTraitConst, Feature2DTrait};
use opencv::videoio::{VideoCapture, CAP_ANY};
use opencv::prelude::{VideoCaptureTrait, VideoCaptureTraitConst};
use opencv::{imgcodecs, imgproc};

// should only be for usize indexes, not entire Mats
pub struct Stabilizer<T> {
    time_window: Duration,
    start_time: Option<Instant>,
    queue: VecDeque<T>,
    current_stabilized: Option<T>,
}

impl<T> Default for Stabilizer<T> {
    fn default() -> Self {
        Self {
            time_window: Duration::from_millis(2000),
            start_time: None,
            queue: VecDeque::default(),
            current_stabilized: None,
        }
    }
}

impl<T> Stabilizer<T> {
    pub fn new(time_window: Duration) -> Self {
        Self {
            time_window: time_window,
            start_time: None,
            queue: VecDeque::default(),
            current_stabilized: None,
        }
    }
    pub fn put(&mut self, item: T) {
        if self.current_stabilized.is_some() {
            // the queue has stopped expanding

            // add + pop item from the queue
            // calculate the current? or only do that on get()?
            return;
        }

        let start_time = self.start_time.unwrap_or_else(|| Instant::now());
        if start_time.elapsed() > self.time_window {
            // stop expanding the queue
        }

        todo!("add unstabilized item to the buffer") 
    }
    pub fn get(&self) -> Option<&T> {
        self.current_stabilized.as_ref()
    }
}

// Don't implement this. just do .next() once on `kebab::device`.
//pub fn capture_one(device: usize) -> Mat {
//    todo!("open a video device, but only to get the first frame")
//}

struct Features {
    pub(crate) kps: Vector<KeyPoint>,
    pub(crate) dcs: Mat,
    pub(crate) algorithm: Ptr<BRISK>,
}

impl Default for Features {
    fn default() -> Self {
        let brisk = BRISK::create(
            30,  // thresh: i32,
            3,   // octaves: i32,
            1.0, // pattern_scale: f32
        ).unwrap();
        Self {
            kps: Vector::default(),
            dcs: Mat::default(),
            algorithm: brisk,
        }
    }
}

impl Features {
    fn calculate(&mut self, img: &Mat) -> Result<()> {
        let mask = Mat::default();
        self.algorithm.detect_and_compute(
            img,
            &mask,
            &mut self.kps,
            &mut self.dcs,
            false,
        )?;
        Ok(())
    }
}

pub struct Frame {
    img: Mat,
    fts: UnsafeCell<Option<Features>>,
}

impl Frame {
    fn from_img(img: Mat) -> Self {
        Self {
            img: img,
            fts: UnsafeCell::new(None),
        }
    }
    pub(crate) fn features(&self) -> Result<&Features> {
        let fts = unsafe { & *self.fts.get() };
        if let Some(fts) = fts {
            return Ok(&fts);
        };
        let mut tmp = Features::default();
        tmp.calculate(&self.img)?;
        let fts = unsafe { &mut *self.fts.get() };
        *fts = Some(tmp);
        Ok(fts.as_ref().unwrap())
    }
}

pub struct Matches<'a> {
    matches: Vector<DMatch>,
    train_frame: &'a Frame,
    query_frame: &'a Frame,
}

pub fn matches<'a>(train: &'a Frame, query: &'a Frame, threshold: f32) -> Matches<'a> {
    todo!("copy-paste from yenndo vision")    
}

pub fn show(frame: &Frame) {
    todo!("copy-paste from yenndo vision")    
}

pub fn draw_features(frame: &Frame) -> Frame {
    todo!("copy-paste from yenndo vision")    
}

pub fn draw_matches(matches: &Matches) -> Frame {
    todo!("copy-paste from yenndo vision")    
}


pub fn device(index: i32) -> impl Iterator<Item = Frame> {
    std::iter::from_generator(move || {
        let mut cap = VideoCapture::new(index, CAP_ANY).unwrap();
        let Ok(is_opened) = cap.is_opened() else {
            eprintln!("device({}) -> cap.is_opened() -> Err", index);
            return;
        };
        if !is_opened {
            eprintln!("device({}) -> cap.is_opened() -> false", index);
            return;
        }
        loop {
            let mut img = Mat::default();
            let _ = cap.read(&mut img);
            let Ok(img_size) = img.size() else {
                eprintln!("file({}) -> img.size() -> Err", index);
                return;
            };
            if img_size.width == 0 || img_size.height == 0 {
                break;
            }
            yield Frame::from_img(img);
        }
    })
}

pub fn file(path: &str) -> impl Iterator<Item = Frame> {
    let path = path.to_string();
    std::iter::from_generator(move || {
        let Ok(mut cap) = VideoCapture::from_file(&path, CAP_ANY) else {
            eprintln!("file({}) -> VideoCapture::from_file() -> Err", path);
            return;
        };
        let Ok(is_opened) = cap.is_opened() else {
            eprintln!("file({}) -> cap.is_opened() -> Err", path);
            return;
        };
        if !is_opened {
            let img = imgcodecs::imread(&path, imgcodecs::IMREAD_COLOR).unwrap();
            let Ok(img) = img_to_gray(img) else {
                eprintln!("file({}) -> img_to_gray() -> Err", path);
                return;
            };
            yield Frame::from_img(img);
            return;
        }
        loop {
            let mut img = Mat::default();
            let _ = cap.read(&mut img);
            let Ok(img_size) = img.size() else {
                eprintln!("file({}) -> img.size() -> Err", path);
                return;
            };
            if img_size.width == 0 || img_size.height == 0 {
                break;
            }
            let Ok(img) = img_to_gray(img) else {
                eprintln!("file({}) -> img_to_gray() -> Err", path);
                return;
            };
            yield Frame::from_img(img);
        }
    })
}

fn img_to_gray(img: Mat) -> Result<Mat> {
    let mut gray_img = Mat::default();
    imgproc::cvt_color(&img, &mut gray_img, imgproc::COLOR_BGR2GRAY, 0)?;
    Ok(gray_img)
}



//pub fn features(mat: Mat) -> BriskFeatures {
//    todo!("return BRISK features for the mat")
//}
//
//pub fn most_similar(frame: BriskFeatures, options: &[BriskFeatures]) -> usize {
//    todo!("return most similar index")
//}
//
//pub fn show(mat: Mat) {
//
//}

// and also something for an image/video display loop.
// (at least an image display)



