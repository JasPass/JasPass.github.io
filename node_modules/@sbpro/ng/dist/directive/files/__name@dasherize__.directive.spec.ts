import { Component, DebugElement } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';

import { <%= classify(name) %>Directive } from './<%= dasherize(name) %>.directive';

@Component({
    template: '<div <%= prefix %><%= classify(name) %> param="test"></div>',
})
class TestComponent {}

describe('<%= classify(name) %>Directive', () => {
    let fixture: ComponentFixture<TestComponent>;

    let testComponent: TestComponent;
    let testComponentDE: DebugElement;
    let testComponentNE: Element;

    beforeEach(() => {
        fixture = TestBed.configureTestingModule({
            declarations: [<%= classify(name) %>Directive, TestComponent],
        }).createComponent(TestComponent);

        fixture.detectChanges();

        testComponent = fixture.componentInstance;
        testComponentDE = fixture.debugElement;
        testComponentNE = testComponentDE.nativeElement;
    });

    it('should have param set to test', () => {
        const directiveInstance = testComponentDE.query(By.directive(<%= classify(name) %>Directive));
        expect(directiveInstance.attributes.param).toBe('test');
    });
});
