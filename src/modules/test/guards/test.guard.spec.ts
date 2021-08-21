import { TestBed } from '@angular/core/testing';

import { TestGuard } from './test.guard';

describe('Test Guards', () => {
    let testGuard: TestGuard;

    beforeEach(() => {
        TestBed.configureTestingModule({
            imports: [],
            providers: [TestGuard],
        });
        testGuard = TestBed.inject(TestGuard);
    });

    describe('canActivate', () => {
        it('should return an Observable<boolean>', () => {
            testGuard.canActivate().subscribe(response => {
                expect(response).toEqual(true);
            });
        });
    });

});
