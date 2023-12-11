#![feature(generators)]
#![feature(iter_from_generator)]

use std::time::{Duration, Instant};
use std::collections::VecDeque;
use anyhow::{Result, bail, ensure, anyhow};
use std::cell::UnsafeCell;

use opencv::features2d::{self, BRISK, DrawMatchesFlags};
use opencv::core::{Mat, Vector, KeyPoint, Ptr, DMatch, Scalar, NORM_HAMMING};
use opencv::prelude::{MatTraitConst, Feature2DTrait};
use opencv::videoio::{VideoCapture, CAP_ANY};
use opencv::prelude::{VideoCaptureTrait, VideoCaptureTraitConst};
use opencv::{imgcodecs, imgproc, highgui};
use opencv::prelude::DescriptorMatcherTraitConst;

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

        // implement this whole thing as simply as possible:
        // - queue size determined by how many frames can
        //   be read and stuffed into the queue in X seconds.
        // - return the item most frequently occuring in the
        //   current queue.
        //
        // - T = &Frame (we do pointer comparison)
        todo!("add unstabilized item to the buffer") 
    }
    pub fn get(&self) -> Option<&T> {
        self.current_stabilized.as_ref()
    }
}

pub struct Features {
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
    fn calculate(&mut self, img: &Mat) {
        let mask = Mat::default();
        self.algorithm.detect_and_compute(
            img,
            &mask,
            &mut self.kps,
            &mut self.dcs,
            false,
        ).unwrap();
    }
}

pub struct Frame {
    mat: Mat,
    fts: UnsafeCell<Option<Features>>,
}

impl Frame {
    fn from_mat(mat: Mat) -> Self {
        Self {
            mat: mat,
            fts: UnsafeCell::new(None),
        }
    }
    pub(crate) fn features(&self) -> &Features {
        let fts = unsafe { & *self.fts.get() };
        if let Some(fts) = fts {
            return fts;
        };
        let mut tmp = Features::default();
        tmp.calculate(&self.mat);
        let fts = unsafe { &mut *self.fts.get() };
        *fts = Some(tmp);
        fts.as_ref().unwrap()
    }
}

pub fn features(frame: &Frame) -> &Features {
    frame.features()
}

pub struct Matches<'a> {
    matches: Vector<DMatch>,
    train: &'a Frame,
    query: &'a Frame,
}

impl<'a> Matches<'a> {
    pub fn len(&self) -> usize {
        self.matches.len()
    }
}

pub fn matches<'a>(train: &'a Frame, query: &'a Frame, threshold: f32) -> Matches<'a> {
//fn brute_force_match(query_dcs: &Mat, train_dcs: &Mat, threshold: f32) -> Result<Vector<DMatch>> {
    let mut all_matches = Vector::new();
    let mut good_matches = Vector::new();

    // KNN=2 Match
    let bf_matcher = opencv::features2d::BFMatcher::create(NORM_HAMMING, false).unwrap();
    bf_matcher.knn_train_match(
        &query.features().dcs,
        &train.features().dcs,
        &mut all_matches,
        2,
        &Mat::default(),
        false,
    ).unwrap();

    // Filter
    for pair_dmatch in all_matches {
        let m = pair_dmatch.get(0).expect("pair_dmatch.get(0)");
        let n = pair_dmatch.get(1).expect("pair_dmatch.get(1)");
        if m.distance < threshold * n.distance {
            good_matches.push(m);
        }
    }
    Matches {
        matches: good_matches,
        train: train,
        query: query,
    }
}

pub fn window(window: &str) {
    highgui::named_window(window, highgui::WINDOW_NORMAL).unwrap();
}

pub fn show(frame: &Frame, window: &str) {
    if let Err(_) = highgui::imshow(window, &frame.mat) {
        eprintln!("show(): window does not exist: {}", window)
    }
}

pub fn close(window: &str) {
    let _ = highgui::destroy_window(window);
}

pub fn wait_key(code: i32) -> bool {
    highgui::wait_key(1).unwrap() == code
}

pub fn draw_features(frame: &Frame) -> Frame {
//pub fn draw_keypoints(img: &Mat, kps: &Vector<KeyPoint>) -> Result<Mat> {
    let mut out = Mat::default();
    features2d::draw_keypoints(
        &frame.mat,
        &frame.features().kps,
        &mut out,
        Scalar::all(-1.0),
        DrawMatchesFlags::DEFAULT,
    ).unwrap();
    Frame::from_mat(out)
}

pub fn draw_matches(matches: &Matches) -> Frame {
    let Matches { matches, query, train } = matches;

    let mut out = Mat::default();
    features2d::draw_matches(
        &query.mat,
        &query.features().kps,
        &train.mat,
        &query.features().kps,
        &matches, 
        &mut out,
        Scalar::all(-1.0),
        Scalar::all(-1.0),
        &Vector::default(),
        features2d::DrawMatchesFlags::DEFAULT,
    ).unwrap();
    Frame::from_mat(out)
}

pub fn device(index: i32) -> impl Iterator<Item=Frame> {
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
            yield Frame::from_mat(img);
        }
    })
}

pub fn file(path: &str) -> impl Iterator<Item=Frame> {
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
            yield Frame::from_mat(img);
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
            yield Frame::from_mat(img);
        }
    })
}

fn img_to_gray(img: Mat) -> Result<Mat> {
    let mut gray_img = Mat::default();
    imgproc::cvt_color(&img, &mut gray_img, imgproc::COLOR_BGR2GRAY, 0)?;
    Ok(gray_img)
}


