import { TestBed } from '@angular/core/testing';

import { TestService } from './test.service';

describe('TestService', () => {
    let testService: TestService;

    beforeEach(() => {
        TestBed.configureTestingModule({
            providers: [TestService],
        });
        testService = TestBed.inject(TestService);
    });

    describe('getTest$', () => {
        it('should return Observable<Test>', () => {
            expect(testService).toBeDefined();
        });
    });
});
