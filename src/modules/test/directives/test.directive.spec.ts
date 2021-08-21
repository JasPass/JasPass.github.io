import { Component, DebugElement } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';

import { TestDirective } from './test.directive';

@Component({
    template: '<div sbTest param="test"></div>',
})
class TestComponent {}

describe('TestDirective', () => {
    let fixture: ComponentFixture<TestComponent>;

    let testComponent: TestComponent;
    let testComponentDE: DebugElement;
    let testComponentNE: Element;

    beforeEach(() => {
        fixture = TestBed.configureTestingModule({
            declarations: [TestDirective, TestComponent],
        }).createComponent(TestComponent);

        fixture.detectChanges();

        testComponent = fixture.componentInstance;
        testComponentDE = fixture.debugElement;
        testComponentNE = testComponentDE.nativeElement;
    });

    it('should have param set to test', () => {
        const directiveInstance = testComponentDE.query(By.directive(TestDirective));
        expect(directiveInstance.attributes.param).toBe('test');
    });
});
