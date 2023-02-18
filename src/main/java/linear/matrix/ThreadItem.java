package linear.matrix;

import java.util.concurrent.Semaphore;

public class ThreadItem implements Runnable {
    Runnable fn;
    final private Semaphore lockIn = new Semaphore(0, true);
    final private Semaphore lockOut = new Semaphore(0, true);

    public void setFn(Runnable fn) {
        this.fn = fn;
        this.resume();
    }

    public void run() {
        do {
            this.pause();

            if (this.fn == null) {
                break;
            }

            this.fn.run();
        } while (true);

        this.lockIn.release();
    }

    public void prepare() {
        try {
            this.lockOut.acquire();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    public void resume() {
        this.lockIn.release();
    }

    public void waitTask() {
        try {
            this.lockOut.acquire();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    private void pause() {
        this.lockOut.release();
        try {
            this.lockIn.acquire();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}
